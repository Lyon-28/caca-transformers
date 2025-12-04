#!/bin/bash

echo "🚀 Setting up Caca Transformers project..."

# Create directory structure
mkdir -p caca_transformers tests examples docs .github/workflows

# Create __init__.py files
touch caca_transformers/__init__.py
touch tests/__init__.py

echo "📁 Directory structure created"

# Create virtual environment
python -m venv venv
source venv/bin/activate

echo "🔧 Installing dependencies..."
pip install --upgrade pip
pip install build twine wheel

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy your code into the respective files"
echo "2. Run: pip install -e '.[dev]'"
echo "3. Run tests: pytest tests/"
echo "4. Build: python -m build"
echo "5. Upload to TestPyPI: twine upload --repository testpypi dist/*"