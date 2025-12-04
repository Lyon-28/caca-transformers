"""Caca model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CacaConfig(PretrainedConfig):
    r"""
    Configuration class for Caca models.
    
    This is the configuration class to store the configuration of a :class:`~CacaModel` or a
    :class:`~CacaForCausalLM`. It is used to instantiate a Caca model according to the specified
    arguments, defining the model architecture.
    
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to
    control the model outputs. Read the documentation from :class:`~transformers.PretrainedConfig`
    for more information.
    
    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 100000):
            Vocabulary size of the model.
        hidden_size (:obj:`int`, `optional`, defaults to 2048):
            Dimensionality of the hidden representations.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (:obj:`int`, `optional`, defaults to 4):
            Number of key-value heads for Grouped Query Attention (GQA).
        intermediate_size (:obj:`int`, `optional`, defaults to 5632):
            Dimensionality of the "intermediate" (often called FFN) layer.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 8192):
            Maximum sequence length that this model might ever be used with.
        rms_norm_eps (:obj:`float`, `optional`, defaults to 1e-6):
            The epsilon used by the RMS normalization layers.
        rope_theta (:obj:`float`, `optional`, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        hidden_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the hidden states.
        sliding_window (:obj:`int`, `optional`, defaults to 4096):
            Sliding window attention size. If None, full attention is used.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        use_flash_attn (:obj:`bool`, `optional`, defaults to False):
            Whether to use Flash Attention for faster computation.
        use_cache (:obj:`bool`, `optional`, defaults to True):
            Whether the model should return the last key/values attentions.
    
    Example:
        >>> from caca_transformers import CacaConfig, CacaForCausalLM
        >>> 
        >>> # Initializing a Caca-1B configuration
        >>> configuration = CacaConfig()
        >>> 
        >>> # Initializing a model from the configuration
        >>> model = CacaForCausalLM(configuration)
        >>> 
        >>> # Accessing the model configuration
        >>> configuration = model.config
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
        **kwargs,
    ):
        # Validation
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
        """Validate configuration parameters"""
        import warnings
        
        # Basic validations
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got: {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got: {hidden_size}")
        if num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be > 0, got: {num_hidden_layers}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be > 0, got: {num_attention_heads}")
        if num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be > 0, got: {num_key_value_heads}")
        
        # Head dimension validation
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        
        # GQA validation
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads}) for GQA"
            )
        
        head_dim = hidden_size // num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")
        
        # Warnings for suboptimal configurations
        if head_dim < 32:
            warnings.warn(f"head_dim ({head_dim}) is very small. Recommended minimum: 64")
        
        if sliding_window is not None and sliding_window > max_position_embeddings:
            raise ValueError(
                f"sliding_window ({sliding_window}) cannot be larger than "
                f"max_position_embeddings ({max_position_embeddings})"
            )