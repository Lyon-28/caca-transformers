# 🚀 Caca Transformers

<div align="center">

[![PyPI version](https://badge.fury.io/py/caca-transformers.svg)](https://badge.fury.io/py/caca-transformers)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/caca-transformers)](https://pepy.tech/project/caca-transformers)

**Modern Transformer Architecture with GQA, RoPE, SwiGLU & Flash Attention**

[Installation](#installation) • [Quick Start](#quick-start) • [Models](#models) • [Documentation](#documentation) • [Examples](#examples)

</div>

---

## ✨ Features

- **🔄 Grouped Query Attention (GQA)** - Optimal balance between speed and quality
- **🌀 RoPE (Rotary Positional Embeddings)** - Better extrapolation for long sequences
- **⚡ SwiGLU Activation** - Superior performance in language modeling
- **📊 RMSNorm** - More efficient and stable than LayerNorm
- **🪟 Sliding Window Attention** - Memory efficient for long contexts
- **💫 Flash Attention Support** - Up to 4x faster attention (optional)
- **🔄 Multi-Backend Support** - Flash Attention → xFormers → SDPA → Standard
- **💾 KV Cache** - Efficient autoregressive generation
- **🎯 30+ Model Variants** - From 10M to 1T parameters

---

## 📦 Installation

### Basic Installation

```bash
pip install caca-transformers
```

### With Optimization Backends

```bash
# With xFormers (3x speedup - recommended)
pip install caca-transformers[xformers]

# With all optimizations
pip install caca-transformers[all]

# For training
pip install caca-transformers[training]

# For development
pip install caca-transformers[dev]
```

### From Source

```bash
git clone https://github.com/Lyon-28/caca-transformers.git
cd caca-transformers
pip install -e .
```

---

## 🚀 Quick Start

### Basic Usage

```python
from caca_transformers import CacaForCausalLM, CacaConfig

# Create model from config
config = CacaConfig(
    vocab_size=50000,
    hidden_size=2048,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=4,
)
model = CacaForCausalLM(config)

# Or use a predefined variant
from caca_transformers import create_caca_model
model, config = create_caca_model("caca-1B")

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Load from Hugging Face

```python
from transformers import AutoModelForCausalLM

# Load pretrained model
model = AutoModelForCausalLM.from_pretrained(
    "Caca-AI/caca-1b",
    trust_remote_code=True
)
```

### Simple Inference

```python
import torch

model.eval()
input_ids = torch.randint(0, 50000, (1, 128))

with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    logits = outputs['logits']
    
print(f"Output shape: {logits.shape}")  # [1, 128, vocab_size]
```

### Generation with KV Cache

```python
with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        use_cache=True,  # Important for speed!
    )
```

---

## 🏗️ Available Models

### Model Variants

We provide 30+ model variants ranging from 10M to 1T parameters:

| Category | Models | Use Case |
|----------|--------|----------|
| **Tiny/Small** | 10M, 50M, 100M, 250M, 500M | Edge devices, rapid experimentation |
| **Medium** | 1B, 2B, 3B, 5B, 7B, 8B, 10B | Production applications, fine-tuning |
| **Large** | 13B, 20B, 30B, 40B, 50B, 60B, 70B, 100B | Research, high-performance apps |
| **Extra Large** | 150B, 200B, 300B, 400B, 500B | Advanced research |
| **Ultra** | 600B, 700B, 800B, 900B, 1000B | Frontier AI research |

### Quick Comparison

```python
from caca_transformers import compare_variants

compare_variants(["caca-1B", "caca-7B", "caca-70B"])
```

Output:
```
Variant         Params       Hidden   Layers   Heads    KV Heads   Memory (FP16)
---------------------------------------------------------------------------------
caca-1B           1.0B       2048     24       16       4          2.0 GB
caca-7B           7.0B       4096     48       32       8          14.0 GB
caca-70B         70.0B       8192     96       64       8          140.0 GB
```

### List All Variants

```python
from caca_transformers import list_all_variants

list_all_variants()
```

---

## 📚 Documentation

### Configuration

```python
from caca_transformers import CacaConfig

config = CacaConfig(
    vocab_size=100000,              # Vocabulary size
    hidden_size=2048,                # Hidden dimension
    num_hidden_layers=24,            # Number of layers
    num_attention_heads=16,          # Query heads
    num_key_value_heads=4,           # KV heads (for GQA)
    intermediate_size=5632,          # FFN intermediate size
    max_position_embeddings=8192,    # Max sequence length
    sliding_window=4096,             # Sliding window size
    rope_theta=10000.0,              # RoPE base frequency
    rms_norm_eps=1e-6,               # RMSNorm epsilon
    attention_dropout=0.0,           # Attention dropout
    hidden_dropout=0.0,              # Hidden dropout
    use_flash_attn=False,            # Enable Flash Attention
    use_cache=True,                  # Enable KV cache
)
```

### Training Example

```python
from transformers import Trainer, TrainingArguments
from caca_transformers import CacaForCausalLM

model = CacaForCausalLM.from_pretrained("Caca-AI/caca-1b")

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    warmup_steps=1000,
    weight_decay=0.1,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=100,
    save_steps=1000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=your_dataset,
)

trainer.train()
```

### Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Or use 8-bit/4-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = CacaForCausalLM.from_pretrained(
    "Caca-AI/caca-7b",
    quantization_config=quantization_config,
    device_map="auto",
)
```

---

## ⚡ Performance

### Attention Backends

The model automatically selects the best available backend:

| Backend | Speedup | Installation |
|---------|---------|--------------|
| **Flash Attention** | 4x | `pip install flash-attn --no-build-isolation` |
| **xFormers** | 3x | `pip install xformers` |
| **PyTorch SDPA** | 2x | Built-in (PyTorch 2.0+) |
| **Standard** | 1x | Built-in |

### Benchmark Results

```python
# Run benchmark
from examples.benchmark import benchmark_forward

model, config = create_caca_model("caca-1B")
model = model.cuda().eval()

input_ids = torch.randint(0, 50000, (1, 512)).cuda()
elapsed = benchmark_forward(model, input_ids, num_iterations=100)

print(f"Throughput: {(100 * 512) / elapsed:.0f} tokens/sec")
```

---

## 🎯 Use Cases

### 1. Research & Experimentation

```python
# Quick experimentation with small models
model, config = create_caca_model("caca-100M")

# Test architecture modifications
config.sliding_window = 2048
config.num_key_value_heads = 2
```

### 2. Custom Pretraining

```python
from caca_transformers import create_caca_model
from transformers import Trainer

# Create untrained model
model, config = create_caca_model("caca-1B")

# Pretrain on your domain-specific data
trainer = Trainer(
    model=model,
    train_dataset=your_domain_dataset,
    args=training_args,
)
trainer.train()
```

### 3. Fine-tuning

```python
# Load pretrained model
model = CacaForCausalLM.from_pretrained("Caca-AI/caca-1b")

# Fine-tune for specific task
training_args = TrainingArguments(
    learning_rate=1e-5,  # Lower LR for fine-tuning
    num_train_epochs=3,
    # ... other args
)

trainer.train()
```

### 4. Instruction Tuning

```python
# Prepare instruction dataset
from datasets import load_dataset

dataset = load_dataset("your-instruction-dataset")

# Fine-tune with instruction format
# ... training code
```

---

## 📖 Examples

### Complete Training Script

See [examples/pretrain.py](https://github.com/Lyon-28/caca-transformers/blob/main/examples/pretrain.py)

```bash
python examples/pretrain.py \
    --variant caca-1B \
    --dataset wikitext \
    --output_dir ./output \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 8
```

### Inference Script

See [examples/inference.py](https://github.com/Lyon-28/caca-transformers/blob/main/examples/inference.py)

```bash
python examples/inference.py \
    --model Caca-AI/caca-1b \
    --prompt "Hello, I am Caca" \
    --max_length 100 \
    --temperature 0.7
```

### Benchmark Script

See [examples/benchmark.py](https://github.com/Lyon-28/caca-transformers/blob/main/examples/benchmark.py)

```bash
python examples/benchmark.py \
    --variant caca-1B \
    --batch_size 1 \
    --seq_length 512 \
    --num_iterations 100
```

---

## 🔧 Advanced Usage

### Custom Architecture

```python
from caca_transformers import CacaConfig, CacaForCausalLM

# Create custom architecture
config = CacaConfig(
    vocab_size=50000,
    hidden_size=1536,           # Custom hidden size
    num_hidden_layers=20,        # Custom depth
    num_attention_heads=12,      # Custom heads
    num_key_value_heads=3,       # 4:1 GQA ratio
    intermediate_size=4096,      # Custom FFN size
    sliding_window=2048,         # Custom window
)

model = CacaForCausalLM(config)
```

### Multi-GPU Training

```python
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# Training loop
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### DeepSpeed Integration

```bash
# Create DeepSpeed config (ds_config.json)
# Then run:
deepspeed train.py --deepspeed ds_config.json
```

---

## 🛠️ Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=caca_transformers --cov-report=html
```

### Code Formatting

```bash
# Format code
black caca_transformers/
isort caca_transformers/

# Check formatting
black --check caca_transformers/
flake8 caca_transformers/
```

---

## 📊 Model Comparison

### vs LLaMA 2

| Feature | Caca | LLaMA 2 |
|---------|------|---------|
| GQA | ✅ | ✅ |
| RoPE | ✅ | ✅ |
| SwiGLU | ✅ | ✅ |
| RMSNorm | ✅ | ✅ |
| Sliding Window | ✅ | ❌ |
| Flash Attention | ✅ (optional) | ✅ |
| Open Weights | ✅ (coming) | ✅ |

### vs Mistral

| Feature | Caca | Mistral |
|---------|------|---------|
| GQA | ✅ | ✅ |
| Sliding Window | ✅ | ✅ |
| RoPE | ✅ | ✅ |
| Model Sizes | 30+ variants | Few variants |

---

## ⚠️ Known Limitations

1. **Untrained Weights**: All models are released with random initialization
2. **No Tokenizer**: Users need to provide their own tokenizer
3. **Flash Attention**: Optional and may require specific CUDA setup
4. **Large Models**: 70B+ models require multi-GPU setup

### Workarounds

```python
# Use tokenizer from another model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Disable Flash Attention if issues
config = CacaConfig(use_flash_attn=False)

# Start with smaller models for testing
model, config = create_caca_model("caca-100M")
```

---

## 💬 Community & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Lyon-28/caca-transformers/issues)
- **GitHub Discussions**: [Ask questions & share ideas](https://github.com/Lyon-28/caca-transformers/discussions)
- **Hugging Face**: [@Caca-AI](https://huggingface.co/Caca-AI)
- **Email**: cacatransformers@gmail.com

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/Lyon-28/caca-transformers/blob/main/CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star the repo if you find it useful!

---

## 📄 License

Apache License 2.0 - See [LICENSE](https://github.com/Lyon-28/caca-transformers/blob/main/LICENSE) for details.

Free for commercial and research use.

---

## 📚 Citation

If you use Caca Transformers in your research, please cite:

```bibtex
@software{caca2025,
  author = {Lyon},
  title = {Caca Transformers: Modern Transformer Architecture with GQA, RoPE, and SwiGLU},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Lyon-28/caca-transformers},
  note = {Apache License 2.0}
}
```

---

## 🙏 Acknowledgments

Inspired by research from:
- **LLaMA** (Meta AI) - GQA architecture
- **Mistral** (Mistral AI) - Sliding Window Attention
- **GPT-NeoX** (EleutherAI) - RoPE implementation
- **PaLM** (Google) - SwiGLU activation
- **Flash Attention** (Dao et al.) - Efficient attention

---

## 🔗 Links

- **GitHub**: [Lyon-28/caca-transformers](https://github.com/Lyon-28/caca-transformers)
- **PyPI**: [caca-transformers](https://pypi.org/project/caca-transformers/)
- **Hugging Face (Main)**: [@Lyon28](https://huggingface.co/Lyon28)
- **Hugging Face (Org)**: [@Caca-AI](https://huggingface.co/Caca-AI)
- **Documentation**: [Coming Soon]

---

<div align="center">

**Made with ❤️ for the AI Community**

[![Star on GitHub](https://img.shields.io/github/stars/Lyon-28/caca-transformers?style=social)](https://github.com/Lyon-28/caca-transformers)

</div>