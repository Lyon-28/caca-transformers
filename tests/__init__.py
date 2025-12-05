"""
Caca Transformers - Arsitektur Transformer Modern
==================================================

Fitur:
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- SwiGLU Activation
- RMSNorm
- Sliding Window Attention
- Flash Attention Support

Contoh penggunaan:
    >>> from caca_transformers import CacaForCausalLM, CacaConfig
    >>> config = CacaConfig()
    >>> model = CacaForCausalLM(config)
    
    # Atau load dari pretrained
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
    get_variant_info,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__url__",
    "__license__",
    "CacaConfig",
    "CacaModel",
    "CacaForCausalLM",
    "CacaPreTrainedModel",
    "CacaAttention",
    "CacaMLP",
    "CacaDecoderLayer",
    "CacaRMSNorm",
    "CacaRotaryEmbedding",
    "create_caca_model",
    "CACA_VARIANTS",
    "estimate_training_tokens",
    "estimate_batch_size",
    "estimate_gpu_recommendation",
    "estimate_training_time",
    "list_all_variants",
    "compare_variants",
    "get_variant_info",
    "get_available_backends",
    "print_backend_info",
]


def _register_for_auto_class():
    """Register model dengan transformers AutoClasses"""
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        
        try:
            existing = AutoConfig.for_model("caca")
            return
        except KeyError:
            pass
        
        AutoConfig.register("caca", CacaConfig)
        AutoModel.register(CacaConfig, CacaModel)
        AutoModelForCausalLM.register(CacaConfig, CacaForCausalLM)
        
    except ImportError:
        pass
    except Exception:
        pass


_register_for_auto_class()

def get_available_backends():
    """
    Cek backend attention optimization yang tersedia.
    
    Returns:
        dict: Dictionary dengan informasi backend
        
    Contoh:
        >>> from caca_transformers import get_available_backends
        >>> backends = get_available_backends()
        >>> print(backends)
        {'flash_attn': True, 'xformers': False, 'sdpa': True, 'cuda': True}
    """
    backends = {}
    
    try:
        import torch
        backends['pytorch'] = True
        backends['cuda'] = torch.cuda.is_available()
        backends['pytorch_version'] = torch.__version__
    except ImportError:
        backends['pytorch'] = False
        backends['cuda'] = False
        backends['pytorch_version'] = None
    
    try:
        import flash_attn
        backends['flash_attn'] = True
        backends['flash_attn_version'] = flash_attn.__version__
    except ImportError:
        backends['flash_attn'] = False
        backends['flash_attn_version'] = None
    
    try:
        import xformers
        backends['xformers'] = True
        backends['xformers_version'] = xformers.__version__
    except ImportError:
        backends['xformers'] = False
        backends['xformers_version'] = None
    
    try:
        import torch.nn.functional as F
        backends['sdpa'] = hasattr(F, 'scaled_dot_product_attention')
    except:
        backends['sdpa'] = False
    
    return backends


def print_backend_info():
    """
    Print informasi tentang backend optimization yang tersedia.
    
    Contoh:
        >>> from caca_transformers import print_backend_info
        >>> print_backend_info()
    """
    backends = get_available_backends()
    
    print("\n🔧 Caca Transformers - Informasi Backend")
    print("=" * 50)
    
    if backends['pytorch']:
        print(f"PyTorch: ✅ {backends['pytorch_version']}")
    else:
        print("PyTorch: ❌ Tidak terinstall")
        print("  Install: pip install torch>=2.0.0")
        return
    
    if backends['cuda']:
        print("CUDA: ✅ Tersedia")
    else:
        print("CUDA: ⚠️  Tidak tersedia (CPU only)")
    
    if backends['flash_attn']:
        print(f"Flash Attention: ✅ {backends['flash_attn_version']}")
    else:
        print("Flash Attention: ❌ Tidak terinstall (opsional)")
        print("  Install: pip install flash-attn --no-build-isolation")
    
    if backends['xformers']:
        print(f"xFormers: ✅ {backends['xformers_version']}")
    else:
        print("xFormers: ❌ Tidak terinstall (recommended)")
        print("  Install: pip install xformers")
    
    if backends['sdpa']:
        print("PyTorch SDPA: ✅ Tersedia")
    else:
        print("PyTorch SDPA: ❌ Tidak tersedia")
        print("  Upgrade PyTorch: pip install torch>=2.0.0")
    
    print()
    if backends['flash_attn']:
        print("Backend utama: Flash Attention (4x speedup)")
    elif backends['xformers']:
        print("Backend utama: xFormers (3x speedup)")
    elif backends['sdpa']:
        print("Backend utama: PyTorch SDPA (2x speedup)")
    else:
        print("Backend utama: Standard Attention (baseline)")
        print("⚠️  Pertimbangkan install xFormers untuk performa lebih baik")
    
    print()


def _check_critical_dependencies():
    """Cek dependensi critical (silent kecuali error)"""
    try:
        import torch
    except ImportError:
        import warnings
        warnings.warn(
            "PyTorch tidak ditemukan! Caca Transformers memerlukan PyTorch.\n"
            "Install dengan: pip install torch>=2.0.0\n"
            "Kunjungi: https://pytorch.org/get-started/locally/",
            ImportWarning,
            stacklevel=2
        )


_check_critical_dependencies()
