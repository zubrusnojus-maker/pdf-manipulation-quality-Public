# Contributing to PDF Processing Toolkit

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pdf-processing.git
   cd pdf-processing
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- poppler-utils (for PDF rasterization)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### IDE Setup

For VS Code, recommended extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Ruff (charliermarsh.ruff)

## Code Style

We follow these conventions:

### Python Style

- **PEP 8** for general style
- **Ruff** for linting and formatting
- **Type hints** for all function signatures
- **Docstrings** for all public functions/classes (Google style)

```python
def process_pdf(
    input_path: str | Path,
    output_path: str | Path,
    operation: Literal["summarize", "rewrite"] = "summarize",
) -> Path:
    """
    Process a PDF file with the specified operation.

    Args:
        input_path: Path to the input PDF file
        output_path: Path to save the output PDF
        operation: The text manipulation operation to perform

    Returns:
        Path to the created output file

    Raises:
        PDFValidationError: If the input file is invalid
        PDFProcessingError: If processing fails
    """
```

### Formatting

Run before committing:

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_pdf_redactor.py -v

# Run tests matching a pattern
pytest -k "test_redact" -v

# Run only unit tests
pytest -m unit

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Tests go in the `tests/` directory
- Name test files `test_<module>.py`
- Name test functions `test_<description>`
- Use fixtures from `conftest.py`
- Mock external dependencies (ML models, file I/O)

Example:

```python
import pytest
from pdf_redactor import PDFRedactor

class TestPDFRedactor:
    """Tests for PDFRedactor class."""

    def test_detect_email(self) -> None:
        """Should detect email addresses in text."""
        redactor = PDFRedactor()
        text = "Contact: john@example.com"

        detections = redactor.detect_sensitive_data(text)

        emails = [d for d in detections if "email" in d[1]]
        assert len(emails) == 1
        assert "john@example.com" in emails[0][0]

    def test_redact_pdf_creates_output(
        self, sensitive_data_pdf: Path, temp_dir: Path
    ) -> None:
        """Should create redacted output file."""
        output = temp_dir / "output.pdf"
        redactor = PDFRedactor(verify_redaction=False)

        stats = redactor.redact_pdf(sensitive_data_pdf, output)

        assert output.exists()
        assert stats.total_replacements > 0
```

### Coverage Requirements

- Minimum 80% code coverage
- All new code must have tests
- Critical paths require 100% coverage

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest main:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run all checks**:
   ```bash
   ruff format .
   ruff check .
   pytest --cov=. --cov-fail-under=80
   ```

3. **Update documentation** if needed

### PR Guidelines

- **Title**: Clear, concise description (e.g., "Add email pattern to redactor")
- **Description**: Explain what and why, not how
- **Link issues**: Use "Fixes #123" or "Closes #123"
- **Small PRs**: Prefer focused, single-purpose changes
- **Tests**: Include tests for new functionality

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe tests added/modified

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. Automated checks must pass (tests, lint, security)
2. At least one maintainer approval required
3. All review comments addressed
4. Branch is up to date with main

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Sample input (if possible, without sensitive data)

### Feature Requests

Include:
- Clear description of the feature
- Use case/motivation
- Proposed implementation (optional)
- Alternatives considered

### Security Issues

For security vulnerabilities, please email directly instead of creating a public issue.

## Architecture Guidelines

### Module Structure

- `constants.py` - All configuration values
- `pdf_utils.py` - Shared utility functions
- `pdf_processor.py` - PDF processing pipeline
- `pdf_redactor.py` - Data redaction tool
- `quick_pdf_process.py` - Simple interface

### Adding New Features

1. Add constants to `constants.py`
2. Add utility functions to `pdf_utils.py`
3. Implement feature in appropriate module
4. Add comprehensive tests
5. Update documentation

### Error Handling

- Use custom exceptions (`PDFValidationError`, `PDFProcessingError`)
- Log errors with appropriate levels
- Provide helpful error messages
- Clean up resources in `finally` blocks

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for contributing!
