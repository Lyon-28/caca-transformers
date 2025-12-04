"""
Example script for pretraining Caca models.

Usage:
    python examples/pretrain.py --variant caca-1B --dataset your-dataset
"""

import argparse
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from caca_transformers import CacaForCausalLM, create_caca_model


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Caca models")
    parser.add_argument(
        "--variant",
        type=str,
        default="caca-1B",
        help="Model variant to train (e.g., caca-1B, caca-7B)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name from HuggingFace",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Warmup steps",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Enable Flash Attention",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"🚀 Pretraining {args.variant}")
    print(f"{'='*60}\n")
    
    # Create model
    print(f"📦 Creating {args.variant}...")
    model, config = create_caca_model(args.variant)
    
    if args.use_flash_attn:
        config.use_flash_attn = True
        print("⚡ Flash Attention enabled")
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("💾 Gradient checkpointing enabled")
    
    # Load dataset
    print(f"📚 Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, args.dataset_config)
    
    # Simple tokenization (you should use a proper tokenizer)
    # This is just for demonstration
    def tokenize_function(examples):
        # You need to implement proper tokenization here
        # This is a placeholder
        return {"input_ids": examples["text"]}
    
    print("🔤 Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=None,  # You need a tokenizer here
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=1.0,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb" if args.use_wandb else "none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        data_collator=data_collator,
    )
    
    # Train
    print("\n🏋️ Starting training...\n")
    trainer.train()
    
    # Save
    print(f"\n💾 Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()