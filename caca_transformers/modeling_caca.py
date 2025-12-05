"""Model Caca dengan arsitektur transformer modern."""

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

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class CacaRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings=8192, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len, position_offset=0):
        t = torch.arange(position_offset, position_offset + seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        return cos.to(x.dtype), sin.to(x.dtype)

def rotate_half(x):

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):

    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CacaAttention(nn.Module):

    _backend_logged = False
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.sliding_window = config.sliding_window

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = CacaRotaryEmbedding(
            self.head_dim, 
            config.max_position_embeddings, 
            config.rope_theta
        )
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        self.has_flash_attn = HAS_FLASH_ATTN and config.use_flash_attn
        self.has_xformers = HAS_XFORMERS
        self.has_sdpa = HAS_SDPA
        self._flash_attn_failed = False
        self._xformers_warned = False
        
        self._mask_cache = {}
        self._max_cache_size = 10

    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        past_key_value=None,
        use_cache=False,
    ):
        batch_size, seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            position_offset = past_key_value[0].shape[-2]
        else:
            position_offset = 0
        
        cos, sin = self.rotary_emb(query_states, seq_length, position_offset)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        kv_seq_len = key_states.shape[-2]

        if self.has_flash_attn and not self._flash_attn_failed and attention_mask is None:
            if (query_states.device.type == 'cuda' and 
                query_states.dtype in [torch.float16, torch.bfloat16]):
                try:
                    attn_output = self._flash_attention(query_states, key_states, value_states)
                except Exception as e:
                    if not self._flash_attn_failed:
                        logger.warning(f"Flash Attention gagal: {e}, fallback ke backend lain")
                        self._flash_attn_failed = True
                    attn_output = self._fallback_attention(query_states, key_states, value_states, attention_mask, kv_seq_len)
            else:
                attn_output = self._fallback_attention(query_states, key_states, value_states, attention_mask, kv_seq_len)
        else:
            attn_output = self._fallback_attention(query_states, key_states, value_states, attention_mask, kv_seq_len)

        attn_output = self.o_proj(attn_output)
        return attn_output, present_key_value
    
    def _flash_attention(self, query_states, key_states, value_states):

        batch_size, num_heads, seq_length, head_dim = query_states.shape
        kv_seq_len = key_states.shape[-2]
        
        original_dtype = query_states.dtype

        if original_dtype not in [torch.float16, torch.bfloat16]:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = original_dtype

        query_states = query_states.transpose(1, 2).contiguous().to(compute_dtype)
        key_states = key_states.transpose(1, 2).contiguous().to(compute_dtype)
        value_states = value_states.transpose(1, 2).contiguous().to(compute_dtype)
        
        if self.sliding_window is not None and self.sliding_window < kv_seq_len:
            window_size = (self.sliding_window, 0)
        else:
            window_size = (-1, 0)
            
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            causal=True,
            window_size=window_size,
        )
        
        if self._flash_attn_failed:
            logger.info("Flash Attention kembali normal")
            self._flash_attn_failed = False
        
        attn_output = attn_output.to(original_dtype)
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        return attn_output
    
    def _fallback_attention(self, query_states, key_states, value_states, attention_mask, kv_seq_len):

        device_type = query_states.device.type
        
        if self.has_xformers and device_type == 'cuda' and attention_mask is None:
            try:
                return self._xformers_attention(query_states, key_states, value_states, kv_seq_len)
            except Exception as e:
                logger.warning(f"⚠️ xFormers gagal: {e}, fallback ke SDPA")
        
        if self.has_sdpa:
            return self._sdpa_attention(query_states, key_states, value_states, attention_mask, kv_seq_len)
            
            return self._standard_attention(query_states, key_states, value_states, attention_mask, kv_seq_len)
    
    def _create_causal_mask(self, query_length, key_length, dtype, device):

        cache_key = (query_length, key_length, device, self.sliding_window)
        
        if cache_key in self._mask_cache:
            cached_mask = self._mask_cache[cache_key]

            if cached_mask.dtype != dtype:
                return cached_mask.to(dtype)
            return cached_mask
            
        if query_length > key_length:
            raise ValueError(
                f"query_length ({query_length}) > key_length ({key_length})"
            )
            
        query_pos = torch.arange(query_length, device=device) + (key_length - query_length)
        key_pos = torch.arange(key_length, device=device)
        distance = query_pos[:, None] - key_pos[None, :]
        
        mask = distance < 0
        if self.sliding_window is not None:
            too_far_mask = distance > self.sliding_window
            mask = mask | too_far_mask
            
        float_mask = torch.zeros(1, 1, query_length, key_length, dtype=dtype, device=device)
        float_mask.masked_fill_(mask, torch.finfo(dtype).min)
        
        if len(self._mask_cache) < self._max_cache_size:
            self._mask_cache[cache_key] = float_mask
            
        return float_mask
    
    def _xformers_attention(self, query_states, key_states, value_states, kv_seq_len):

        batch_size, num_heads, seq_length, head_dim = query_states.shape
        
        attn_bias = self._create_causal_mask(seq_length, kv_seq_len, query_states.dtype, query_states.device)
        
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

        batch_size, num_heads, seq_length, head_dim = query_states.shape
        
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_length, kv_seq_len, query_states.dtype, query_states.device)
        
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        return attn_output
        
    def _standard_attention(self, query_states, key_states, value_states, attention_mask, kv_seq_len):

        batch_size, num_heads, seq_length, head_dim = query_states.shape
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_length, kv_seq_len, attn_weights.dtype, attn_weights.device)
        
        attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        
        if self.training and not self.config.use_cache:
            del attn_weights
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return attn_output

class CacaMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)
        return output

class CacaDecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attn = CacaAttention(config)
        self.mlp = CacaMLP(config)
        self.input_layernorm = CacaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = CacaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states, 
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value

class CacaPreTrainedModel(PreTrainedModel):

    config_class = CacaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CacaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
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
        if attention_mask is None:
            return None
        
        batch_size, seq_length = input_shape
        
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask[:, None, :, :]
        
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
            raise TypeError(f"input_ids harus torch.Tensor, dapat {type(input_ids)}")
        
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids harus 2D [batch_size, seq_length], "
                f"dapat {input_ids.dim()}D dengan shape {input_ids.shape}"
            )
            
        if input_ids.dtype not in [torch.long, torch.int, torch.int32, torch.int64]:
            raise TypeError(
                f"input_ids harus integer dtype, dapat {input_ids.dtype}. "
                f"Gunakan input_ids.long() untuk convert."
            )
            
        if (input_ids < 0).any():
            min_val = input_ids.min().item()
            raise ValueError(f"input_ids mengandung nilai negatif: {min_val}")
            
        if (input_ids >= self.config.vocab_size).any():
            max_val = input_ids.max().item()
            raise ValueError(
                f"input_ids mengandung nilai >= vocab_size. "
                f"Max value: {max_val}, vocab_size: {self.config.vocab_size}"
            )
            
        batch_size, seq_length = input_ids.shape
        if seq_length > self.config.max_position_embeddings:
            logger.warning(
                f"Sequence length ({seq_length}) > max_position_embeddings "
                f"({self.config.max_position_embeddings})"
            )
            
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        hidden_states = self.embed_tokens(input_ids)
        
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(
                attention_mask, 
                (batch_size, seq_length), 
                hidden_states.dtype
            )
            
        if use_cache and past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        
        present_key_values = [] if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
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
        
        hidden_states = self.norm(hidden_states)
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": tuple(present_key_values) if use_cache else None,
        }

    def _gradient_checkpointing_forward(self, layer, hidden_states, attention_mask):
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
            output = (logits,) + tuple(v for v in outputs.values() if v is not None)
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
