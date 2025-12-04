"""Utility functions for Caca models"""

from typing import Dict, List, Optional, Tuple
import warnings


# Import untuk create_caca_model
def _lazy_import_config_and_model():
    """Lazy import to avoid circular dependency"""
    from .configuration_caca import CacaConfig
    from .modeling_caca import CacaForCausalLM
    return CacaConfig, CacaForCausalLM


CACA_VARIANTS = {
    
    # TINY MODELS (10M - 100M)
    "caca-10M": {
        "vocab_size": 100000,
        "hidden_size": 256,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 688,
        "max_position_embeddings": 8192,
        "sliding_window": 1024,
    },
    
    "caca-50M": {
        "vocab_size": 100000,
        "hidden_size": 512,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 1376,
        "max_position_embeddings": 8192,
        "sliding_window": 2048,
    },
    
    "caca-100M": {
        "vocab_size": 100000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 3,
        "intermediate_size": 2048,
        "max_position_embeddings": 8192,
        "sliding_window": 2048,
    },
    
    # SMALL MODELS (250M - 500M)
    "caca-250M": {
        "vocab_size": 100000,
        "hidden_size": 896,
        "num_hidden_layers": 20,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 2432,
        "max_position_embeddings": 8192,
        "sliding_window": 2048,
    },
    
    "caca-500M": {
        "vocab_size": 100000,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 2752,
        "max_position_embeddings": 8192,
        "sliding_window": 4096,
    },
    
    # MEDIUM MODELS (1B - 10B)
    "caca-1B": {
        "vocab_size": 100000,
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 5632,
        "max_position_embeddings": 8192,
        "sliding_window": 4096,
    },
    
    "caca-2B": {
        "vocab_size": 100000,
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "num_key_value_heads": 5,
        "intermediate_size": 6912,
        "max_position_embeddings": 8192,
        "sliding_window": 4096,
    },
    
    "caca-3B": {
        "vocab_size": 100000,
        "hidden_size": 3072,
        "num_hidden_layers": 32,
        "num_attention_heads": 24,
        "num_key_value_heads": 6,
        "intermediate_size": 8192,
        "max_position_embeddings": 8192,
        "sliding_window": 4096,
    },
    
    "caca-5B": {
        "vocab_size": 100000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "max_position_embeddings": 8192,
        "sliding_window": 4096,
    },
    
    "caca-7B": {
        "vocab_size": 100000,
        "hidden_size": 4096,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "max_position_embeddings": 8192,
        "sliding_window": 4096,
    },
    
    "caca-8B": {
        "vocab_size": 100000,
        "hidden_size": 4096,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "max_position_embeddings": 8192,
        "sliding_window": 4096,
    },
    
    "caca-10B": {
        "vocab_size": 100000,
        "hidden_size": 5120,
        "num_hidden_layers": 48,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "intermediate_size": 13824,
        "max_position_embeddings": 8192,
        "sliding_window": 4096,
    },
    
    # LARGE MODELS (13B - 100B)
    "caca-13B": {
        "vocab_size": 128000,
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "intermediate_size": 13824,
        "max_position_embeddings": 16384,
        "sliding_window": 8192,
    },
    
    "caca-20B": {
        "vocab_size": 128000,
        "hidden_size": 6144,
        "num_hidden_layers": 64,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "intermediate_size": 16384,
        "max_position_embeddings": 16384,
        "sliding_window": 8192,
    },
    
    "caca-30B": {
        "vocab_size": 128000,
        "hidden_size": 7168,
        "num_hidden_layers": 64,
        "num_attention_heads": 56,
        "num_key_value_heads": 8,
        "intermediate_size": 19200,
        "max_position_embeddings": 16384,
        "sliding_window": 8192,
    },
    
    "caca-40B": {
        "vocab_size": 128000,
        "hidden_size": 8192,
        "num_hidden_layers": 72,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 22016,
        "max_position_embeddings": 16384,
        "sliding_window": 8192,
    },
    
    "caca-50B": {
        "vocab_size": 128000,
        "hidden_size": 8192,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 22016,
        "max_position_embeddings": 32768,
        "sliding_window": 16384,
    },
    
    "caca-60B": {
        "vocab_size": 128000,
        "hidden_size": 8192,
        "num_hidden_layers": 88,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 22016,
        "max_position_embeddings": 32768,
        "sliding_window": 16384,
    },
    
    "caca-70B": {
        "vocab_size": 128000,
        "hidden_size": 8192,
        "num_hidden_layers": 96,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 22016,
        "max_position_embeddings": 32768,
        "sliding_window": 16384,
    },
    
    "caca-100B": {
        "vocab_size": 128000,
        "hidden_size": 10240,
        "num_hidden_layers": 96,
        "num_attention_heads": 80,
        "num_key_value_heads": 8,
        "intermediate_size": 27648,
        "max_position_embeddings": 32768,
        "sliding_window": 16384,
    },
    
    # EXTRA LARGE MODELS (150B - 500B)
    "caca-150B": {
        "vocab_size": 256000,
        "hidden_size": 12288,
        "num_hidden_layers": 96,
        "num_attention_heads": 96,
        "num_key_value_heads": 12,
        "intermediate_size": 32768,
        "max_position_embeddings": 32768,
        "sliding_window": 16384,
    },
    
    "caca-200B": {
        "vocab_size": 256000,
        "hidden_size": 12288,
        "num_hidden_layers": 120,
        "num_attention_heads": 96,
        "num_key_value_heads": 12,
        "intermediate_size": 32768,
        "max_position_embeddings": 32768,
        "sliding_window": 16384,
    },
    
    "caca-300B": {
        "vocab_size": 256000,
        "hidden_size": 14336,
        "num_hidden_layers": 128,
        "num_attention_heads": 112,
        "num_key_value_heads": 14,
        "intermediate_size": 38400,
        "max_position_embeddings": 65536,
        "sliding_window": 32768,
    },
    
    "caca-400B": {
        "vocab_size": 256000,
        "hidden_size": 16384,
        "num_hidden_layers": 128,
        "num_attention_heads": 128,
        "num_key_value_heads": 16,
        "intermediate_size": 44032,
        "max_position_embeddings": 65536,
        "sliding_window": 32768,
    },
    
    "caca-500B": {
        "vocab_size": 256000,
        "hidden_size": 16384,
        "num_hidden_layers": 144,
        "num_attention_heads": 128,
        "num_key_value_heads": 16,
        "intermediate_size": 44032,
        "max_position_embeddings": 65536,
        "sliding_window": 32768,
    },
    
    # ULTRA MODELS (600B - 1T)
    "caca-600B": {
        "vocab_size": 256000,
        "hidden_size": 18432,
        "num_hidden_layers": 144,
        "num_attention_heads": 144,
        "num_key_value_heads": 18,
        "intermediate_size": 49152,
        "max_position_embeddings": 65536,
        "sliding_window": 32768,
    },
    
    "caca-700B": {
        "vocab_size": 256000,
        "hidden_size": 19456,
        "num_hidden_layers": 148,
        "num_attention_heads": 152,
        "num_key_value_heads": 19,
        "intermediate_size": 52224,
        "max_position_embeddings": 65536,
        "sliding_window": 32768,
    },
    
    "caca-800B": {
        "vocab_size": 256000,
        "hidden_size": 20480,
        "num_hidden_layers": 152,
        "num_attention_heads": 160,
        "num_key_value_heads": 20,
        "intermediate_size": 55296,
        "max_position_embeddings": 131072,
        "sliding_window": 65536,
    },
    
    "caca-900B": {
        "vocab_size": 256000,
        "hidden_size": 21504,
        "num_hidden_layers": 156,
        "num_attention_heads": 168,
        "num_key_value_heads": 21,
        "intermediate_size": 58368,
        "max_position_embeddings": 131072,
        "sliding_window": 65536,
    },
    
    "caca-1000B": {
        "vocab_size": 256000,
        "hidden_size": 24576,
        "num_hidden_layers": 160,
        "num_attention_heads": 192,
        "num_key_value_heads": 24,
        "intermediate_size": 65536,
        "max_position_embeddings": 131072,
        "sliding_window": 65536,
    },
}


def create_caca_model(variant_name: str) -> Tuple:
    """
    Create a Caca model from variant name.
    
    Args:
        variant_name: Name of the variant (e.g., "caca-1B", "caca-7B")
        
    Returns:
        Tuple of (model, config)
        
    Raises:
        ValueError: If variant_name is not found
        
    Example:
        >>> from caca_transformers import create_caca_model
        >>> model, config = create_caca_model("caca-1B")
        >>> print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    """
    if variant_name not in CACA_VARIANTS:
        available = ", ".join(sorted(CACA_VARIANTS.keys()))
        raise ValueError(
            f"❌ Variant '{variant_name}' not found!\n"
            f"Available variants: {available}"
        )

    # Lazy import to avoid circular dependency
    CacaConfig, CacaForCausalLM = _lazy_import_config_and_model()
    
    config = CacaConfig(**CACA_VARIANTS[variant_name])
    model = CacaForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Model {variant_name} created successfully!")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"💾 Size: {total_params/1e6:.1f}M / {total_params/1e9:.2f}B")
    print(f"🔧 Trainable params: {trainable_params:,}")

    return model, config


def estimate_parameters(config: Dict) -> int:
    """
    Estimate number of parameters from config dictionary.
    
    Args:
        config: Configuration dictionary with model hyperparameters
        
    Returns:
        Estimated number of parameters
    """
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    num_layers = config['num_hidden_layers']
    intermediate_size = config['intermediate_size']
    num_heads = config['num_attention_heads']
    num_kv_heads = config['num_key_value_heads']
    
    # Embedding
    embed_params = vocab_size * hidden_size
    
    # Per layer
    head_dim = hidden_size // num_heads
    
    # Attention: Q, K, V, O projections
    attn_params = (
        hidden_size * num_heads * head_dim +      # Q
        hidden_size * num_kv_heads * head_dim +   # K
        hidden_size * num_kv_heads * head_dim +   # V
        hidden_size * hidden_size                  # O
    )
    
    # FFN: gate, up, down (SwiGLU)
    ffn_params = (
        hidden_size * intermediate_size +  # gate
        hidden_size * intermediate_size +  # up
        intermediate_size * hidden_size    # down
    )
    
    # RMSNorm (2 per layer: pre-attn and pre-ffn)
    norm_params = hidden_size * 2
    
    layer_params = attn_params + ffn_params + norm_params
    total_layer_params = layer_params * num_layers
    
    # Final norm
    final_norm_params = hidden_size
    
    # Total (assuming tied embeddings, so LM head not counted separately)
    total_params = embed_params + total_layer_params + final_norm_params
    
    return total_params


def estimate_training_tokens(total_params: float) -> str:
    """
    Estimate recommended training tokens based on model size.
    
    Args:
        total_params: Total number of parameters
        
    Returns:
        String representation of recommended tokens (e.g., "20B", "1.5T")
        
    Example:
        >>> estimate_training_tokens(1e9)
        '10B'
    """
    if total_params < 100e6:
        # Tiny models: 5x tokens
        tokens = total_params * 5
        if tokens < 1e9:
            return f"{tokens/1e6:.0f}M"
        else:
            return f"{tokens/1e9:.1f}B"
    
    elif total_params < 1e9:
        # Small models: 10x tokens
        tokens = total_params * 10
        return f"{tokens/1e9:.1f}B"
    
    elif total_params < 10e9:
        # Medium models: 20x tokens
        tokens = total_params * 20
        return f"{tokens/1e9:.0f}B"
    
    elif total_params < 100e9:
        # Large models: 40x tokens
        tokens = total_params * 40
        if tokens < 1e12:
            return f"{tokens/1e9:.0f}B"
        else:
            return f"{tokens/1e12:.1f}T"
    
    else:
        # Extra large models: 50x tokens
        tokens = total_params * 50
        return f"{tokens/1e12:.1f}T"


def estimate_batch_size(total_params: float) -> str:
    """
    Estimate recommended batch size based on model size.
    
    Args:
        total_params: Total number of parameters
        
    Returns:
        String representation of recommended batch size range
        
    Example:
        >>> estimate_batch_size(1e9)
        '8-32'
    """
    if total_params < 50e6:
        return "64-256"
    elif total_params < 100e6:
        return "32-128"
    elif total_params < 500e6:
        return "16-64"
    elif total_params < 1e9:
        return "8-32"
    elif total_params < 3e9:
        return "4-16"
    elif total_params < 7e9:
        return "2-8"
    elif total_params < 30e9:
        return "1-4"
    elif total_params < 70e9:
        return "1-2"
    else:
        return "1"


def estimate_gpu_recommendation(total_params: float) -> str:
    """
    Estimate GPU requirements based on model size.
    
    Args:
        total_params: Total number of parameters
        
    Returns:
        String describing recommended GPU configuration
        
    Example:
        >>> estimate_gpu_recommendation(7e9)
        'RTX 3090 / RTX 4090 / RTX A5000'
    """
    # Estimate memory for training (includes optimizer states, gradients, etc.)
    # Rule of thumb: ~16 bytes per parameter for mixed precision training
    memory_training = total_params * 16 / 1e9
    
    if memory_training < 8:
        return "GTX 1060 6GB+ / RTX 2060 / RTX 3060"
    elif memory_training < 12:
        return "RTX 2080 Ti / RTX 3060 12GB"
    elif memory_training < 24:
        return "RTX 3090 / RTX 4090 / RTX A5000"
    elif memory_training < 40:
        return "RTX 6000 Ada / A5000 / A40"
    elif memory_training < 80:
        return "A100 40GB / A6000 48GB"
    elif memory_training < 160:
        return "A100 80GB / H100 80GB"
    elif memory_training < 320:
        return "2x A100 80GB"
    elif memory_training < 640:
        return "4x A100 80GB"
    elif memory_training < 1280:
        return "8x A100 80GB"
    else:
        num_gpus = int(memory_training / 80) + 1
        return f"Multi-node cluster ({num_gpus}+ A100 80GB)"


def estimate_training_time(
    total_params: float,
    tokens: str,
    num_gpus: int = 1,
    gpu_type: str = "A100"
) -> str:
    """
    Estimate training time based on model size and configuration.
    
    Args:
        total_params: Total number of parameters
        tokens: Number of training tokens (e.g., "20B", "1T")
        num_gpus: Number of GPUs
        gpu_type: Type of GPU (A100, H100, A6000, RTX4090, RTX3090)
        
    Returns:
        String describing estimated training time
        
    Example:
        >>> estimate_training_time(7e9, "100B", num_gpus=8, gpu_type="A100")
        '5.2 days'
    """
    # Parse tokens string to float
    if isinstance(tokens, str):
        tokens_str = tokens.upper().replace('T', 'e12').replace('B', 'e9').replace('M', 'e6')
        tokens_val = float(tokens_str)
    else:
        tokens_val = tokens
    
    # Calculate FLOPs (6N rule for forward+backward)
    flops_per_token = 6 * total_params
    total_flops = flops_per_token * tokens_val
    
    # GPU TFLOPS (peak FP16/BF16)
    gpu_tflops = {
        "A100": 312,
        "H100": 500,
        "A6000": 154,
        "RTX4090": 165,
        "RTX3090": 71,
    }
    
    tflops = gpu_tflops.get(gpu_type, 312)
    
    # Effective TFLOPS (assume 45% MFU - Model FLOPs Utilization)
    effective_tflops = tflops * 0.45 * num_gpus * 1e12
    
    # Calculate time
    time_seconds = total_flops / effective_tflops
    
    # Format output
    if time_seconds < 3600:
        return f"{time_seconds/60:.0f} minutes"
    elif time_seconds < 86400:
        return f"{time_seconds/3600:.1f} hours"
    elif time_seconds < 604800:
        return f"{time_seconds/86400:.1f} days"
    elif time_seconds < 2592000:
        return f"{time_seconds/604800:.1f} weeks"
    else:
        return f"{time_seconds/2592000:.1f} months"


def list_all_variants() -> None:
    """
    List all available model variants with their specifications.
    
    Example:
        >>> from caca_transformers import list_all_variants
        >>> list_all_variants()
    """
    print("\n" + "="*60)
    print("📚 AVAILABLE CACA MODEL VARIANTS")
    print("="*60 + "\n")
    
    categories = {
        "🐣 Tiny/Small (10M-500M)": [
            "caca-10M", "caca-50M", "caca-100M", "caca-250M", "caca-500M"
        ],
        "🦅 Medium (1B-10B)": [
            "caca-1B", "caca-2B", "caca-3B", "caca-5B", 
            "caca-7B", "caca-8B", "caca-10B"
        ],
        "🦁 Large (13B-100B)": [
            "caca-13B", "caca-20B", "caca-30B", "caca-40B",
            "caca-50B", "caca-60B", "caca-70B", "caca-100B"
        ],
        "🐋 Extra Large (150B-500B)": [
            "caca-150B", "caca-200B", "caca-300B", "caca-400B", "caca-500B"
        ],
        "🦕 Ultra (600B-1T)": [
            "caca-600B", "caca-700B", "caca-800B", "caca-900B", "caca-1000B"
        ],
    }
    
    for category, variants in categories.items():
        print(f"{category}")
        for variant in variants:
            config = CACA_VARIANTS[variant]
            params = estimate_parameters(config)
            print(
                f"  • {variant:15s} - {params/1e9:6.1f}B params - "
                f"{config['hidden_size']:5d}d x {config['num_hidden_layers']:3d} layers"
            )
        print()
    
    print("="*60 + "\n")


def compare_variants(variant_names: List[str]) -> None:
    """
    Compare multiple model variants side-by-side.
    
    Args:
        variant_names: List of variant names to compare
        
    Example:
        >>> from caca_transformers import compare_variants
        >>> compare_variants(["caca-1B", "caca-7B", "caca-70B"])
    """
    print("\n" + "="*60)
    print("📊 VARIANT COMPARISON")
    print("="*60 + "\n")
    
    print(
        f"{'Variant':<15} {'Params':<12} {'Hidden':<8} {'Layers':<8} "
        f"{'Heads':<8} {'KV Heads':<10} {'Memory (FP16)':<15}"
    )
    print("-" * 95)
    
    for variant in variant_names:
        if variant not in CACA_VARIANTS:
            print(f"⚠️  {variant}: Not found")
            continue
        
        config = CACA_VARIANTS[variant]
        params = estimate_parameters(config)
        memory_fp16 = params * 2 / 1e9
        
        print(
            f"{variant:<15} {params/1e9:>6.1f}B      "
            f"{config['hidden_size']:<8} {config['num_hidden_layers']:<8} "
            f"{config['num_attention_heads']:<8} {config['num_key_value_heads']:<10} "
            f"{memory_fp16:>6.1f} GB"
        )
    
    print("\n")


def get_variant_info(variant_name: str) -> Dict:
    """
    Get detailed information about a specific variant.
    
    Args:
        variant_name: Name of the variant
        
    Returns:
        Dictionary with variant information
        
    Raises:
        ValueError: If variant not found
        
    Example:
        >>> from caca_transformers import get_variant_info
        >>> info = get_variant_info("caca-1B")
        >>> print(info['total_params'])
    """
    if variant_name not in CACA_VARIANTS:
        raise ValueError(f"Variant '{variant_name}' not found")
    
    config = CACA_VARIANTS[variant_name]
    params = estimate_parameters(config)
    
    return {
        'variant_name': variant_name,
        'config': config,
        'total_params': params,
        'params_B': params / 1e9,
        'memory_fp16_gb': params * 2 / 1e9,
        'memory_fp32_gb': params * 4 / 1e9,
        'recommended_tokens': estimate_training_tokens(params),
        'recommended_batch_size': estimate_batch_size(params),
        'recommended_gpu': estimate_gpu_recommendation(params),
    }


# Convenience function for testing
if __name__ == "__main__":
    # Test helper functions
    params_1b = 1e9
    
    print("Testing helper functions for 1B model:")
    print(f"  Recommended tokens: {estimate_training_tokens(params_1b)}")
    print(f"  Recommended batch size: {estimate_batch_size(params_1b)}")
    print(f"  Recommended GPU: {estimate_gpu_recommendation(params_1b)}")
    print(f"  Training time (8x A100): {estimate_training_time(params_1b, '20B', num_gpus=8)}")
    
    params_70b = 70e9
    print("\nTesting helper functions for 70B model:")
    print(f"  Recommended tokens: {estimate_training_tokens(params_70b)}")
    print(f"  Recommended batch size: {estimate_batch_size(params_70b)}")
    print(f"  Recommended GPU: {estimate_gpu_recommendation(params_70b)}")
    print(f"  Training time (64x A100): {estimate_training_time(params_70b, '1.4T', num_gpus=64)}")
    
    print("\n" + "="*60)
    list_all_variants()
    
    print("="*60)
    compare_variants(["caca-1B", "caca-7B", "caca-70B"])