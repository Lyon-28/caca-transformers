from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="caca-transformers",
    version="0.1.0",
    author="Lyon",
    author_email="cacatransformers@gmail.com",
    description="Caca: Transformer Architecture with GQA, RoPE, and SwiGLU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lyon-28/caca-transformers",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "safetensors>=0.3.0",
    ],
    extras_require={
        "flash-attn": ["flash-attn>=2.0.0"],
    },
)
