# Contributing to RECAP

Thank you for your interest in contributing to RECAP! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/recap.git
cd recap
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

## Development Workflow

### Branch Naming

- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring

Example: `feature/add-enzyme-context`

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest tests/ -v
   ```

4. Run linting:
   ```bash
   ruff check recap/ tests/
   black recap/ tests/
   ```

5. Commit your changes:
   ```bash
   git commit -m "Add feature: description"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Open a Pull Request

## Code Style

### Python Style

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 88 characters (Black default)
- Use docstrings for all public functions and classes

### Docstring Format

```python
def complete_reaction(
    substrates: List[str],
    products: Optional[List[str]] = None,
) -> CompletionResult:
    """
    Complete a biochemical reaction.
    
    Args:
        substrates: List of substrate SMILES strings.
        products: Optional list of product SMILES strings.
    
    Returns:
        CompletionResult containing predictions and scores.
    
    Raises:
        ValueError: If substrates is empty.
    
    Example:
        >>> result = complete_reaction(["ATP", "glucose"])
        >>> print(result.products)
    """
```

### Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `chore:` Maintenance

Example: `feat: add EC number context to completion`

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_completer.py -v

# With coverage
pytest tests/ --cov=recap --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setups
- Mock external dependencies

## Documentation

### Building Docs

```bash
cd docs
make html
```

### Writing Docs

- Use Markdown for tutorials
- Use docstrings for API documentation
- Include code examples

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add entry to CHANGELOG.md
4. Request review from maintainers
5. Address review comments
6. Squash commits if requested

## Reporting Issues

### Bug Reports

Include:
- Python version
- Package version
- Minimal reproducible example
- Expected vs actual behavior
- Error messages/tracebacks

### Feature Requests

Include:
- Use case description
- Proposed API/interface
- Alternative approaches considered

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn

## Questions?

Open an issue or reach out to the maintainers.

Thank you for contributing! 🎉
