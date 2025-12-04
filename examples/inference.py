"""
Example script for inference with Caca models.

Usage:
    python examples/inference.py --model Caca-AI/caca-1b --prompt "Hello, world!"
"""

import argparse
import torch
from caca_transformers import CacaForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with Caca models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., Caca-AI/caca-1b)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, I am Caca",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"🤖 Caca Inference")
    print(f"{'='*60}\n")
    
    print(f"📦 Loading model: {args.model}...")
    model = CacaForCausalLM.from_pretrained(args.model)
    model = model.to(args.device)
    model.eval()
    
    print(f"✅ Model loaded on {args.device}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Note: You need to implement tokenization
    # This is just a placeholder
    print(f"\n📝 Prompt: {args.prompt}")
    print(f"🎲 Generating {args.num_return_sequences} sequence(s)...\n")
    
    # Placeholder for tokenization
    # input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
    
    # For demonstration, using random tokens
    input_ids = torch.randint(0, 50000, (1, 10)).to(args.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences,
            do_sample=True,
            use_cache=True,
        )
    
    print("Generated sequences:")
    for i, sequence in enumerate(outputs):
        print(f"\n[{i+1}] {sequence.tolist()}")
        # With tokenizer: print(tokenizer.decode(sequence, skip_special_tokens=True))
    
    print(f"\n{'='*60}")
    print("✅ Inference complete!")


if __name__ == "__main__":
    main()