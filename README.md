# 🚀 Caca Transformers

Arsitektur Transformer kustom dengan fitur modern:

- ✅ **Grouped Query Attention (GQA)** - Efisiensi tanpa mengorbankan kualitas
- ✅ **RoPE (Rotary Positional Embeddings)** - Proven positional encoding
- ✅ **SwiGLU Activation** - Superior performance in language modeling
- ✅ **RMSNorm** - Normalisasi efisien
- ✅ **Flash Attention Support** - 2-4x lebih cepat
- ✅ **Sliding Window Attention** - Efisiensi untuk context panjang

## Installation

### From PyPI

```bash
pip install caca-transformers
```

### From GitHub

```bash
pip install git+https://github.com/Lyon-28/caca-transformers.git
```

### With Flash Attention

```bash
pip install caca-transformers[flash-attn]
```

## Quick Start

### Load Pretrained Model

```python
from caca_transformers import CacaForCausalLM

model = CacaForCausalLM.from_pretrained("Lyon28/caca-1B")
```

### With AutoModel

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Lyon28/caca-1B",
    trust_remote_code=True
)
```

## Available Models

### Tiny & Small (10M - 500M)
| Model | Params | Hidden | Layers | Context |
|-------|--------|--------|--------|---------|
| caca-10M | 10M | 256 | 8 | 8K |
| caca-50M | 50M | 512 | 12 | 8K |
| caca-100M | 100M | 768 | 12 | 8K |
| caca-250M | 250M | 896 | 20 | 8K |
| caca-500M | 500M | 1,024 | 24 | 8K |

### Medium (1B - 10B)
| Model | Params | Hidden | Layers | Context |
|-------|--------|--------|--------|---------|
| caca-1B | 1B | 2,048 | 24 | 8K |
| caca-2B | 2B | 2,560 | 32 | 8K |
| caca-3B | 3B | 3,072 | 32 | 8K |
| caca-5B | 5B | 4,096 | 32 | 8K |
| caca-7B | 7B | 4,096 | 48 | 8K |
| caca-10B | 10B | 5,120 | 48 | 8K |

### Large (13B - 100B)
| Model | Params | Hidden | Layers | Context |
|-------|--------|--------|--------|---------|
| caca-13B | 13B | 5,120 | 64 | 16K |
| caca-20B | 20B | 6,144 | 64 | 16K |
| caca-30B | 30B | 7,168 | 64 | 16K |
| caca-50B | 50B | 8,192 | 80 | 32K |
| caca-70B | 70B | 8,192 | 96 | 32K |
| caca-100B | 100B | 10,240 | 96 | 32K |

### Extra Large (150B - 500B)
| Model | Params | Hidden | Layers | Context |
|-------|--------|--------|--------|---------|
| caca-150B | 150B | 12,288 | 96 | 32K |
| caca-200B | 200B | 12,288 | 120 | 32K |
| caca-300B | 300B | 14,336 | 128 | 64K |
| caca-400B | 400B | 16,384 | 128 | 64K |
| caca-500B | 500B | 16,384 | 144 | 64K |

### Ultra (600B - 1T)
| Model | Params | Hidden | Layers | Context |
|-------|--------|--------|--------|---------|
| caca-600B | 600B | 18,432 | 144 | 64K |
| caca-700B | 700B | 20,480 | 144 | 64K |
| caca-800B | 800B | 20,480 | 160 | 128K |
| caca-900B | 900B | 22,528 | 160 | 128K |
| caca-1000B | 1T | 24,576 | 160 | 128K |

> See all models: [https://huggingface.co/Lyon28](https://huggingface.co/Lyon28)

## Training

```python
from transformers import Trainer, TrainingArguments
from caca_transformers import CacaForCausalLM, CacaConfig

config = CacaConfig()
model = CacaForCausalLM(config)

training_args = TrainingArguments(
    output_dir="./caca-model",
    per_device_train_batch_size=8,
    gradient_checkpointing=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

## Fitur

### Flash Attention

Otomatis aktif jika tersedia:

```python
config = CacaConfig(use_flash_attn=True)
```

### Gradient Checkpointing

Untuk training model besar:

```python
model.gradient_checkpointing_enable()
```

### Sliding Window Attention

Untuk efisiensi pada sequence panjang:

```python
config = CacaConfig(sliding_window=4096)
```

## Arsitektur

```
Caca Model
├── Embedding Layer
├── N × Decoder Layers
│   ├── RMSNorm
│   ├── Grouped Query Attention (GQA)
│   │   ├── RoPE Positional Encoding
│   │   └── Flash Attention (optional)
│   ├── RMSNorm
│   └── SwiGLU MLP
└── RMSNorm + LM Head
```

## License

Apache 2.0

## Links

- 🤗 HuggingFace: [https://huggingface.co/Lyon28](https://huggingface.co/Lyon28)
- 📦 GitHub: [https://github.com/Lyon-28/caca-transformers](https://github.com/Lyon-28/o)
- 📚 PyPI: [https://pypi.org/project/caca-transformers/](https://pypi.org/project/caca-transformers/)

## Citation

```bibtex
@software{caca2025,
  author = {Lyon},
  title = {Caca Transformers: Transformer Architecture},
  year = {2025},
  url = {https://github.com/Lyon-28/caca-transformers}
}
```
