"""
Benchmark script for Caca models.

Usage:
    python examples/benchmark.py --variant caca-1B --batch_size 1 --seq_length 512
"""

import argparse
import time
import torch
from caca_transformers import create_caca_model


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Caca models")
    parser.add_argument(
        "--variant",
        type=str,
        default="caca-1B",
        help="Model variant to benchmark",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for benchmark",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=512,
        help="Sequence length",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of iterations for benchmark",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Enable Flash Attention",
    )
    
    return parser.parse_args()


def benchmark_forward(model, input_ids, num_iterations, warmup_iterations, device):
    """Benchmark forward pass"""
    
    # Warmup
    print(f"🔥 Warming up ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = model(input_ids)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"⏱️  Benchmarking ({num_iterations} iterations)...")
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(input_ids)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    
    return elapsed_time


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"📊 Benchmarking {args.variant}")
    print(f"{'='*60}\n")
    
    # Create model
    print(f"📦 Creating model...")
    model, config = create_caca_model(args.variant)
    
    if args.use_flash_attn:
        config.use_flash_attn = True
        print("⚡ Flash Attention enabled")
    
    model = model.to(args.device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded on {args.device}")
    print(f"📊 Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # Create dummy input
    input_ids = torch.randint(
        0, 
        config.vocab_size, 
        (args.batch_size, args.seq_length)
    ).to(args.device)
    
    print(f"\n🔧 Benchmark settings:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sequence length: {args.seq_length}")
    print(f"   Iterations: {args.num_iterations}")
    print()
    
    # Benchmark
    elapsed_time = benchmark_forward(
        model,
        input_ids,
        args.num_iterations,
        args.warmup_iterations,
        args.device
    )
    
    # Calculate metrics
    avg_time = elapsed_time / args.num_iterations
    tokens_per_sec = (args.batch_size * args.seq_length * args.num_iterations) / elapsed_time
    
    print(f"\n{'='*60}")
    print("📈 Results:")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Average per iteration: {avg_time*1000:.2f}ms")
    print(f"Throughput: {tokens_per_sec:.0f} tokens/sec")
    
    if args.device == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak memory: {memory_allocated:.2f}GB")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()