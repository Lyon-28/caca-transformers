"""Konfigurasi model Caca"""

import warnings
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CacaConfig(PretrainedConfig):
    """
    Konfigurasi untuk model Caca.
    
    Args:
        vocab_size: Ukuran vocabulary
        hidden_size: Dimensi hidden state
        num_hidden_layers: Jumlah layer
        num_attention_heads: Jumlah attention heads
        num_key_value_heads: Jumlah KV heads untuk GQA
        intermediate_size: Dimensi FFN intermediate
        max_position_embeddings: Panjang maksimal sequence
        rms_norm_eps: Epsilon untuk RMSNorm
        rope_theta: Base period untuk RoPE
        attention_dropout: Dropout untuk attention
        hidden_dropout: Dropout untuk hidden states
        sliding_window: Ukuran sliding window attention
        initializer_range: Std dev untuk inisialisasi weights
        use_flash_attn: Gunakan Flash Attention
        use_cache: Return KV cache untuk generation
        rope_scaling: Konfigurasi RoPE scaling
        attention_bias: Bias di attention projections
        mlp_bias: Bias di MLP projections
        residual_dropout: Dropout untuk residual connections
        pretraining_tp: Tensor parallelism rank
    """
    
    model_type = "caca"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        vocab_size=100000,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=5632,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        sliding_window=4096,
        initializer_range=0.02,
        use_flash_attn=False,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        rope_scaling=None,
        attention_bias=False,
        mlp_bias=False,
        residual_dropout=0.0,
        pretraining_tp=1,
        **kwargs,
    ):
        self._validate_config(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
            num_key_value_heads, intermediate_size, max_position_embeddings,
            rms_norm_eps, rope_theta, attention_dropout, hidden_dropout,
            sliding_window, initializer_range, bos_token_id, eos_token_id,
            pad_token_id
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.sliding_window = sliding_window
        self.initializer_range = initializer_range
        self.use_flash_attn = use_flash_attn
        self.use_cache = use_cache
        self.head_dim = hidden_size // num_attention_heads
        
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.residual_dropout = residual_dropout
        self.pretraining_tp = pretraining_tp
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    
    def _validate_config(
        self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
        num_key_value_heads, intermediate_size, max_position_embeddings,
        rms_norm_eps, rope_theta, attention_dropout, hidden_dropout,
        sliding_window, initializer_range, bos_token_id, eos_token_id, pad_token_id
    ):
        """Validasi parameter konfigurasi"""
        
        if vocab_size <= 0:
            raise ValueError(f"vocab_size harus > 0, dapat: {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size harus > 0, dapat: {hidden_size}")
        if num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers harus > 0, dapat: {num_hidden_layers}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads harus > 0, dapat: {num_attention_heads}")
        if num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads harus > 0, dapat: {num_key_value_heads}")
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) harus habis dibagi "
                f"num_attention_heads ({num_attention_heads})"
            )
        
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) harus habis dibagi "
                f"num_key_value_heads ({num_key_value_heads}) untuk GQA"
            )
        
        head_dim = hidden_size // num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) harus genap untuk RoPE")
        
        if head_dim < 32:
            warnings.warn(f"head_dim ({head_dim}) sangat kecil. Recommended minimum: 64")
        
        if sliding_window is not None and sliding_window > max_position_embeddings:
            raise ValueError(
                f"sliding_window ({sliding_window}) tidak boleh > "
                f"max_position_embeddings ({max_position_embeddings})"
            )
