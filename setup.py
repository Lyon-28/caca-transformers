"""Setup script for caca-transformers"""

import os
from setuptools import setup, find_packages

# Read version
about = {}
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "caca_transformers", "__version__.py"), "r", encoding="utf-8") as f:
    exec(f.read(), about)

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="caca-transformers",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__email__"],
    description=about["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about["__url__"],
    project_urls={
        "Bug Tracker": "https://github.com/Lyon-28/caca-transformers/issues",
        "Documentation": "https://github.com/Lyon-28/caca-transformers",
        "Source Code": "https://github.com/Lyon-28/caca-transformers",
        "Hugging Face (Main)": "https://huggingface.co/Lyon28",
        "Hugging Face (Org)": "https://huggingface.co/Caca-AI",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "xformers": ["xformers>=0.0.20"],
        "flash-attn": ["flash-attn>=2.0.0"],
        "training": [
            "accelerate>=0.24.0",
            "deepspeed>=0.12.0",
            "bitsandbytes>=0.41.0",
            "wandb>=0.15.0",
        ],
        "all": [
            "xformers>=0.0.20",
            "accelerate>=0.24.0",
            "deepspeed>=0.12.0",
            "bitsandbytes>=0.41.0",
            "wandb>=0.15.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "transformers",
        "pytorch",
        "language-model",
        "deep-learning",
        "machine-learning",
        "gqa",
        "grouped-query-attention",
        "rope",
        "swiglu",
        "flash-attention",
        "llm",
        "causal-lm",
    ],
)