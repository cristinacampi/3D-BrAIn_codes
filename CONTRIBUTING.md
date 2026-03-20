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


### Documentation

- All functions must have docstrings (NumPy style)
- Include parameter descriptions, return values, and examples
- Update docs in `docs/` for significant changes

## Questions?

- Check existing issues/discussions
- Start a discussion on GitHub
- Email: cristina.campi@unige.it


