# Contributing to Caca Transformers

Thank you for your interest in contributing to Caca Transformers! 🎉

## Ways to Contribute

### 🐛 Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, PyTorch version)
- Error messages/stack traces

### 💡 Suggesting Features

Feature requests are welcome! Please:
- Check existing issues first
- Describe the feature and use case
- Explain why it would be useful
- Provide examples if possible

### 📝 Improving Documentation

Documentation improvements are always appreciated:
- Fix typos or unclear explanations
- Add examples
- Improve API documentation
- Write tutorials

### 🔧 Contributing Code

#### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/Lyon-28/caca-transformers.git
cd caca-transformers

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

#### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

```bash
# Format code
black caca_transformers/
isort caca_transformers/

# Check style
flake8 caca_transformers/
```

#### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_modeling.py::TestCacaModeling::test_forward_pass -v

# With coverage
pytest tests/ --cov=caca_transformers --cov-report=html
```

#### Pull Request Process

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make changes** and commit: `git commit -m "Add your feature"`
4. **Run tests**: Make sure all tests pass
5. **Push**: `git push origin feature/your-feature-name`
6. **Open PR**: Create a pull request with clear description

#### Commit Message Guidelines

```
type(scope): brief description

Longer description if needed

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(attention): add xformers backend support
fix(config): validate sliding_window parameter
docs(readme): add installation instructions
test(modeling): add KV cache tests
```

## Development Guidelines

### Code Quality

- Write clear, readable code
- Add docstrings to functions and classes
- Include type hints where appropriate
- Keep functions focused and small
- Avoid unnecessary complexity

### Testing

- Write tests for new features
- Maintain test coverage above 80%
- Test edge cases and error conditions
- Use meaningful test names

### Documentation

- Update README if adding features
- Add docstrings following Google style
- Include code examples
- Update CHANGELOG

## Questions?

- Open a [GitHub Discussion](https://github.com/Lyon-28/caca-transformers/discussions)
- Email: cacatransformers@gmail.com

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and build something great together.

---

Thank you for contributing! 🙏