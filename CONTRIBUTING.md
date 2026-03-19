# Contributing to 3D-BrAIn

Thank you for your interest in contributing to 3D-BrAIn! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on the code, not the person
- Help others learn and grow

## Getting Started

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/your-username/3D-BrAIn_codes.git`
3. **Create** a virtual environment: `conda env create -f environment.yml`
4. **Install** development dependencies: `pip install -e ".[dev]"`

## Development Workflow

### Creating a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### Code Style

Follow PEP 8 guidelines:
```bash
# Format code with black
black .

# Check with flake8
flake8 .

# Type hints recommended
def function_name(param: str) -> int:
    """Function docstring."""
    pass
```

### Documentation

- All functions must have docstrings (NumPy style)
- Include parameter descriptions, return values, and examples
- Update docs in `docs/` for significant changes

Example:
```python
def process_data(input_array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Process input data with threshold filtering.
    
    Parameters
    ----------
    input_array : np.ndarray
        Input data array of shape (n_samples, n_features)
    threshold : float, optional
        Threshold value for filtering. Defaults to 0.5.
    
    Returns
    -------
    np.ndarray
        Processed array of same shape as input
    
    Examples
    --------
    >>> data = np.random.randn(100, 10)
    >>> result = process_data(data, threshold=0.3)
    """
    pass
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. tests/

# Run specific test
pytest tests/test_clustering.py::test_leiden_algo
```

### Committing

```bash
git add .
git commit -m "feat: add new clustering algorithm"
```

Commit message prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `test:` - Test additions
- `chore:` - Build/dependency updates

### Pushing and Pull Requests

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with:
- Clear description of changes
- Reference to related issues
- Before/after comparison if applicable
- Any breaking changes noted

## Reporting Issues

When reporting bugs, include:
- Python version: `python --version`
- Installed packages: `pip freeze`
- Minimal reproducible example
- Full error traceback
- Operating system and hardware info

## Documentation Contributions

- Update docstrings
- Add examples to docs/
- Improve README sections
- Fix typos and clarity

Build docs locally:
```bash
cd docs
../.venv/bin/python -m sphinx -b html source build/html
```

## Performance Considerations

- Use NumPy/SciPy for numerical operations
- Consider GPU acceleration for large datasets
- Profile code with `cProfile` or `line_profiler`
- Test on realistic dataset sizes

## Release Process

(For maintainers)
1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create git tag
4. Build and upload to PyPI

## Questions?

- Check existing issues/discussions
- Start a discussion on GitHub
- Email: cristina.campi@unige.it

Thank you for contributing! 🎉
