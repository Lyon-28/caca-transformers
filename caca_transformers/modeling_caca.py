"""PyTorch Caca model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
from transformers.utils import logging

from .configuration_caca import CacaConfig

logger = logging.get_logger(__name__)

# Check for optional dependencies
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from xformers.ops import memory_efficient_attention
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False

HAS_SDPA = hasattr(F, 'scaled_dot_product_attention')


class CacaRMSNorm(nn.Module):
    """Root Mean Square Normalization"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class CacaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim, max_position_embeddings=8192, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len, position_offset=0):
        """
        Args:
            x: input tensor [batch, num_heads, seq_len, head_dim]
            seq_len: panjang sequence
            position_offset: offset posisi untuk KV cache
        Returns:
            cos, sin: [1, 1, seq_len, head_dim] dengan dtype yang sama dengan x
        """
        # Generate position indices
        t = torch.arange(position_offset, position_offset + seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        
        # Expand untuk broadcast: [1, 1, seq_len, dim]
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        return cos.to(x.dtype), sin.to(x.dtype)

def rotate_half(x):
    """Rotasi setengah dimensi untuk RoPE"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Aplikasikan RoPE ke query dan key
    
    Args:
        q, k: [batch, num_heads, seq_len, head_dim]
        cos, sin: [1, 1, seq_len, head_dim]
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed   


class CacaAttention(nn.Module):
    """Grouped Query Attention dengan multi-backend support"""
    
    _backend_logged = False
    _backend_lock = None
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.sliding_window = config.sliding_window

        # Projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = CacaRotaryEmbedding(
            self.head_dim, 
            config.max_position_embeddings, 
            config.rope_theta
        )
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        # Backend detection
        self.has_flash_attn = HAS_FLASH_ATTN and config.use_flash_attn
        self.has_xformers = HAS_XFORMERS
        self.has_sdpa = HAS_SDPA
        self._flash_attn_failed = False
        
        if not CacaAttention._backend_logged:
            logger.info("=" * 60)
            logger.info("🔧 CacaAttention Backend Configuration")
            logger.info("=" * 60)
            
            if self.has_flash_attn:
                logger.info("✅ Primary: Flash Attention (4x speedup)")
                logger.info("   - Causal masking: native")
                logger.info("   - Sliding window: native")
                logger.info("   - Memory: O(N) instead of O(N²)")
            elif self.has_xformers:
                logger.info("✅ Primary: xFormers (3x speedup)")
                logger.info("   - Causal masking: via attention_bias")
                logger.info("   - Sliding window: via custom mask")
                logger.info("   - Memory: efficient fused kernels")
            elif self.has_sdpa:
                logger.info("✅ Primary: PyTorch SDPA (2x speedup)")
                logger.info("   - Causal masking: via attention_mask")
                logger.info("   - Sliding window: via custom mask")
                logger.info("   - Memory: fused kernel dari PyTorch 2.0+")
            else:
                logger.warning("⚠️  Primary: Standard Attention (baseline)")
                logger.warning("   - No optimization")
                logger.warning("   - Slowest performance")
                logger.warning("   - Recommended: Install xFormers atau upgrade PyTorch")
            
            logger.info("=" * 60)
            
            CacaAttention._backend_logged = True

    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        past_key_value=None,
        use_cache=False,
    ):
        batch_size, seq_length, _ = hidden_states.size()

        # Project ke Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape untuk multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Hitung position offset untuk RoPE (untuk KV cache)
        if past_key_value is not None:
            position_offset = past_key_value[0].shape[-2]  # Panjang KV cache sebelumnya
        else:
            position_offset = 0
             
        # Aplikasikan RoPE - sekarang cos/sin sudah correct shape
        cos, sin = self.rotary_emb(query_states, seq_length, position_offset)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Concatenate dengan past KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Simpan KV cache untuk generation
        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        # Repeat KV heads untuk GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Total sequence length setelah KV cache
        kv_seq_len = key_states.shape[-2]

        # Pilih attention backend
        if self.has_flash_attn and not self._flash_attn_failed and attention_mask is None:
            if (query_states.device.type == 'cuda' and 
                query_states.dtype in [torch.float16, torch.bfloat16]):
                try:
                    attn_output = self._flash_attention(query_states, key_states, value_states)
                except Exception as e:
                    if not self._flash_attn_failed:
                        logger.warning(f"⚠️ Flash Attention gagal: {e}, fallback ke backend lain")
                        self._flash_attn_failed = True
                    attn_output = self._fallback_attention(query_states, key_states, value_states, attention_mask, kv_seq_len)
            else:
                attn_output = self._fallback_attention(query_states, key_states, value_states, attention_mask, kv_seq_len)
        else:
            attn_output = self._fallback_attention(query_states, key_states, value_states, attention_mask, kv_seq_len)

        # Output projection
        attn_output = self.o_proj(attn_output)
        return attn_output, present_key_value
    
    def _flash_attention(self, query_states, key_states, value_states):
        """Flash Attention implementation"""
        batch_size, num_heads, seq_length, head_dim = query_states.shape
        kv_seq_len = key_states.shape[-2]
        
        # Flash Attention expects: [batch, seq_len, num_heads, head_dim]
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()
        
        # Window size: (left_window, right_window)
        # left = berapa token ke belakang bisa diakses
        # right = berapa token ke depan bisa diakses (0 untuk causal)
        if self.sliding_window is not None and self.sliding_window < kv_seq_len:
            window_size = (self.sliding_window, 0)
        else:
            window_size = (-1, 0)  # -1 = unlimited, 0 = causal
        try:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                causal=True,
                window_size=window_size,
            )
            
            if self._flash_attn_failed:
                logger.info("✅ Flash Attention kembali normal")
                self._flash_attn_failed = False
                
        except Exception as e:
            if not self._flash_attn_failed:
                logger.warning(f"⚠️ Flash Attention gagal: {e}")
                logger.warning(f"   Shapes - Q: {query_states.shape}, K: {key_states.shape}, V: {value_states.shape}")
                logger.warning(f"   Device: {query_states.device}, Dtype: {query_states.dtype}")
                logger.warning("   Fallback ke backend lain...")
            
            self._flash_attn_failed = True
            raise 
        
        # Reshape back
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        return attn_output
    
    def _fallback_attention(self, query_states, key_states, value_states, attention_mask, kv_seq_len):
        """Pilih backend terbaik yang tersedia"""
        device_type = query_states.device.type
        
        # Coba xFormers
        if self.has_xformers and device_type == 'cuda' and attention_mask is None:
            try:
                return self._xformers_attention(query_states, key_states, value_states, kv_seq_len)
            except Exception as e:
                logger.warning(f"⚠️ xFormers gagal: {e}, fallback ke SDPA")
        
        # Coba SDPA
        if self.has_sdpa:
            return self._sdpa_attention(query_states, key_states, value_states, attention_mask, kv_seq_len)
        
        # Standard attention
        return self._standard_attention(query_states, key_states, value_states, attention_mask, kv_seq_len)
    
    def _create_causal_mask(self, query_length, key_length, dtype, device):
        """
        Buat causal mask dengan sliding window
        
        Args:
            query_length: panjang query (biasanya 1 untuk generation, N untuk prefill)
            key_length: panjang key (termasuk KV cache)
        Returns:
            mask: [1, 1, query_length, key_length]
        """
        # Buat indices
        query_indices = torch.arange(key_length - query_length, key_length, device=device)
        key_indices = torch.arange(key_length, device=device)
        
        # Causal mask: query token i hanya bisa attend ke key token <= i
        # distance = query_pos - key_pos
        distance = query_indices[:, None] - key_indices[None, :]
        
        # Mask future tokens (distance < 0)
        causal_mask = distance < 0
        
        # Sliding window mask (distance >= window)
        if self.sliding_window is not None:
            window_mask = distance >= self.sliding_window
            causal_mask = causal_mask | window_mask
        
        # Convert to float mask
        mask = torch.zeros(1, 1, query_length, key_length, dtype=dtype, device=device)
        mask.masked_fill_(causal_mask, torch.finfo(dtype).min)
        
        return mask
    
    def _xformers_attention(self, query_states, key_states, value_states, kv_seq_len):
        """xFormers memory efficient attention"""
        batch_size, num_heads, seq_length, head_dim = query_states.shape
        
        # Create causal + sliding window mask
        attn_bias = self._create_causal_mask(seq_length, kv_seq_len, query_states.dtype, query_states.device)
        
        # xFormers expects: [batch, seq_len, num_heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        attn_output = memory_efficient_attention(
            query_states,
            key_states,
            value_states,
            attn_bias=attn_bias,
            p=self.config.attention_dropout if self.training else 0.0,
        )
        
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        return attn_output
    
    def _sdpa_attention(self, query_states, key_states, value_states, attention_mask, kv_seq_len):
        """PyTorch Scaled Dot Product Attention"""
        batch_size, num_heads, seq_length, head_dim = query_states.shape
        
        # Create causal + sliding window mask jika belum ada
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_length, kv_seq_len, query_states.dtype, query_states.device)
        
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=False,  # Karena kita sudah provide mask
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        return attn_output
        
    def _standard_attention(self, query_states, key_states, value_states, attention_mask, kv_seq_len):
        """Standard scaled dot-product attention"""
        batch_size, num_heads, seq_length, head_dim = query_states.shape
        
        # Hitung attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        
        # Apply mask
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_length, kv_seq_len, attn_weights.dtype, attn_weights.device)
        
        attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        
        return attn_output 
    

class CacaMLP(nn.Module):
    """SwiGLU Feedforward Network"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x):
        # SwiGLU: Swish(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)
        return output  
    

class CacaDecoderLayer(nn.Module):
    """Single decoder layer: Attention + FFN dengan residual connections"""
    def __init__(self, config):
        super().__init__()
        self.self_attn = CacaAttention(config)
        self.mlp = CacaMLP(config)
        self.input_layernorm = CacaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = CacaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
        # Self attention dengan residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states, 
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # FFN dengan residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class CacaPreTrainedModel(PreTrainedModel):
    """Base class for all Caca models"""
    config_class = CacaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CacaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        """Initialize weights"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class CacaModel(CacaPreTrainedModel):
    """Model Caca utama (tanpa LM head)"""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([CacaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = CacaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def _prepare_attention_mask(self, attention_mask, input_shape, dtype):
        """Prepare attention mask untuk model"""
        if attention_mask is None:
            return None
        
        batch_size, seq_length = input_shape
        
        # Expand dimensions
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask[:, None, :, :]
        
        # Convert ke format attention (0 = attend, large negative = tidak attend)
        attention_mask = attention_mask.to(dtype=dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
        
        return attention_mask
        
    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        past_key_values=None,
        use_cache=None,
        **kwargs
    ):
        if input_ids is None:
            raise ValueError("input_ids tidak boleh None")
            
        if not torch.is_tensor(input_ids):
            raise TypeError( 
                f"input_ids harus torch.Tensor, dapat {type(input_ids)}"
            )
        
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids harus 2D [batch_size, seq_length], "
                f"dapat {input_ids.dim()}D dengan shape {input_ids.shape}"
            )
            
        if input_ids.dtype not in [torch.long, torch.int, torch.int32, torch.int64]:
            raise TypeError(
                f"input_ids harus integer dtype (torch.long/int), "
                f"dapat {input_ids.dtype}. "
                f"Hint: Gunakan input_ids.long() untuk convert."
            )
            
        if (input_ids < 0).any():
            min_val = input_ids.min().item()
            raise ValueError(
                f"input_ids mengandung nilai negatif: {min_val}. "
                f"Token IDs harus >= 0."
            )
            
        if (input_ids >= self.config.vocab_size).any():
            max_val = input_ids.max().item()
            raise ValueError(
                f"input_ids mengandung nilai >= vocab_size. "
                f"Max value: {max_val}, vocab_size: {self.config.vocab_size}. "
                f"Hint: Pastikan tokenizer vocab_size match dengan model config."
            )
            
        batch_size, seq_length = input_ids.shape
        if seq_length > self.config.max_position_embeddings:
            logger.warning(
                f"⚠️ Sequence length ({seq_length}) > max_position_embeddings "
                f"({self.config.max_position_embeddings}). "
                f"RoPE extrapolation mungkin tidak optimal."
            )
            
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(
                attention_mask, 
                (batch_size, seq_length), 
                hidden_states.dtype
            )
            
        # Initialize past_key_values
        if use_cache and past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        
        present_key_values = [] if use_cache else None
        
        # Loop melalui semua decoder layers
        for idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            # Gradient checkpointing (hanya saat training tanpa KV cache)
            if self.gradient_checkpointing and self.training and past_key_value is None:
                hidden_states = self._gradient_checkpointing_forward(
                    layer,
                    hidden_states,
                    attention_mask,
                )
                
                present_key_value = None
                
            else:
                hidden_states, present_key_value = layer(
                    hidden_states, 
                    attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": tuple(present_key_values) if use_cache else None,
        }

    def _gradient_checkpointing_forward(self, layer, hidden_states, attention_mask):
        """Forward pass dengan gradient checkpointing untuk save memory"""
        from torch.utils.checkpoint import checkpoint
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                hidden_states, attention_mask = inputs
                output, _ = module(
                    hidden_states, 
                    attention_mask, 
                    past_key_value=None,
                    use_cache=False
                )
                return output
            return custom_forward
        
        hidden_states = checkpoint(
            create_custom_forward(layer),
            hidden_states,
            attention_mask,
            use_reentrant=False,
        )
        
        return hidden_states  
    

class CacaForCausalLM(CacaPreTrainedModel, GenerationMixin):
    """Model Caca untuk causal language modeling"""
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = CacaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self, 
        input_ids=None,
        attention_mask=None, 
        labels=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward pass melalui model
        outputs = self.model(
            input_ids, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.get("past_key_values"),
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            
            input_ids = input_ids[:, remove_prefix_length:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past