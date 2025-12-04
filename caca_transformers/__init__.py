"""
Caca Transformers - Modern Transformer Architecture
====================================================

A modern transformer implementation featuring:
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- SwiGLU Activation
- RMSNorm
- Sliding Window Attention
- Flash Attention Support

Example usage:
    >>> from caca_transformers import CacaForCausalLM, CacaConfig
    >>> config = CacaConfig()
    >>> model = CacaForCausalLM(config)
    
    # Or load from pretrained
    >>> model = CacaForCausalLM.from_pretrained("Caca-AI/caca-1b")
"""

from .__version__ import (
    __version__,
    __author__,
    __email__,
    __description__,
    __url__,
    __license__,
)

from .configuration_caca import CacaConfig
from .modeling_caca import (
    CacaModel,
    CacaForCausalLM,
    CacaPreTrainedModel,
    CacaAttention,
    CacaMLP,
    CacaDecoderLayer,
    CacaRMSNorm,
    CacaRotaryEmbedding,
)
from .utils import (
    create_caca_model,
    CACA_VARIANTS,
    estimate_training_tokens,
    estimate_batch_size,
    estimate_gpu_recommendation,
    estimate_training_time,
    list_all_variants,
    compare_variants,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__url__",
    "__license__",
    
    # Configuration
    "CacaConfig",
    
    # Models
    "CacaModel",
    "CacaForCausalLM",
    "CacaPreTrainedModel",
    
    # Components
    "CacaAttention",
    "CacaMLP",
    "CacaDecoderLayer",
    "CacaRMSNorm",
    "CacaRotaryEmbedding",
    
    # Utilities
    "create_caca_model",
    "CACA_VARIANTS",
    "estimate_training_tokens",
    "estimate_batch_size",
    "estimate_gpu_recommendation",
    "estimate_training_time",
    "list_all_variants",
    "compare_variants",
]

# Register with transformers AutoClasses
try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from transformers.utils import logging
    
    logger = logging.get_logger(__name__)
    
    # Register configuration
    AutoConfig.register("caca", CacaConfig)
    
    # Register models
    AutoModel.register(CacaConfig, CacaModel)
    AutoModelForCausalLM.register(CacaConfig, CacaForCausalLM)
    
    logger.info("✅ Caca models registered with Transformers AutoClasses")
    
except ImportError:
    import warnings
    warnings.warn(
        "transformers not found. AutoModel registration skipped. "
        "Install with: pip install transformers>=4.35.0"
    )

# Check for optional dependencies
def _check_optional_dependencies():
    """Check and log available optional dependencies"""
    import warnings
    
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available. Models will run on CPU (much slower).")
    except ImportError:
        warnings.warn("PyTorch not found. Install with: pip install torch>=2.0.0")
    
    # Check for optimization backends
    backends = []
    
    try:
        import flash_attn
        backends.append("Flash Attention")
    except ImportError:
        pass
    
    try:
        import xformers
        backends.append("xFormers")
    except ImportError:
        pass
    
    try:
        import torch.nn.functional as F
        if hasattr(F, 'scaled_dot_product_attention'):
            backends.append("PyTorch SDPA")
    except:
        pass
    
    if backends:
        print(f"✅ Available attention backends: {', '.join(backends)}")
    else:
        warnings.warn(
            "No optimized attention backend found. "
            "Install xFormers for 3x speedup: pip install xformers"
        )

# Run checks on import
_check_optional_dependencies()