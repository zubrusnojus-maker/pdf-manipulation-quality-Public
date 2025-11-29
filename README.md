# PDF Processing Toolkit

A production-ready, quality-optimized toolkit for PDF processing with OCR, text manipulation, and data redaction using state-of-the-art ML models.

[![Tests](https://github.com/YOUR_USERNAME/pdf-processing/actions/workflows/test.yml/badge.svg)](https://github.com/YOUR_USERNAME/pdf-processing/actions/workflows/test.yml)
[![Lint](https://github.com/YOUR_USERNAME/pdf-processing/actions/workflows/lint.yml/badge.svg)](https://github.com/YOUR_USERNAME/pdf-processing/actions/workflows/lint.yml)
[![Security](https://github.com/YOUR_USERNAME/pdf-processing/actions/workflows/security.yml/badge.svg)](https://github.com/YOUR_USERNAME/pdf-processing/actions/workflows/security.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **High-Quality OCR** - TrOCR Large model with 97%+ accuracy
- **Text Manipulation** - Summarize or rewrite documents using Mistral-7B or Llama-13B
- **Data Redaction** - Comprehensive pattern matching for 20+ sensitive data types
- **Privacy-First** - All processing runs locally, no data leaves your machine
- **Production-Ready** - Input validation, error handling, logging, and 80%+ test coverage

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/pdf-processing.git
cd pdf-processing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install poppler (required for PDF rasterization)
# macOS: brew install poppler
# Ubuntu: sudo apt-get install poppler-utils
# Windows: Download from https://poppler.freedesktop.org/
```

### Basic Usage

```bash
# Quick PDF processing (summarization)
python quick_pdf_process.py document.pdf

# Full processing with options
python pdf_processor.py document.pdf -o output.pdf --operation summarize

# Redact sensitive data
python pdf_redactor.py document.pdf -o redacted.pdf
```

## Tools

### pdf_processor.py - Full PDF Processing Pipeline

Process PDFs with OCR and AI-powered text manipulation.

```bash
python pdf_processor.py input.pdf [options]

Options:
  -o, --output        Output file path (default: edited_<input>)
  -m, --model-size    Model size: small (Mistral-7B) or large (Llama-13B)
  --operation         Operation: summarize or rewrite
  --dpi               DPI for rasterization (72-600, default: 300)
  --use-layout        Enable layout analysis for tables
  --optimize          Post-process with pikepdf optimization
  -v, --verbose       Enable verbose logging
```

### pdf_redactor.py - Data Anonymization

Redact sensitive data while preserving document formatting.

```bash
python pdf_redactor.py input.pdf [options]

Options:
  -o, --output        Output file path (default: redacted_<input>)
  --mapping-output    Path to save redaction mappings JSON
  --skip-patterns     Comma-separated patterns to skip (e.g., email,phone_uk)
  --no-verify         Skip post-redaction verification
  --no-metadata       Don't redact document metadata
  --list-patterns     List all available redaction patterns
  -v, --verbose       Enable verbose logging
```

**Detected Patterns:**

- Email addresses, phone numbers (UK/US/International)
- Social Security Numbers, National Insurance Numbers
- Credit card numbers, account numbers, IBANs
- Dates (multiple formats), monetary amounts
- Names with titles, company names
- UK postcodes, US ZIP codes, street addresses
- IP addresses, passport numbers

### quick_pdf_process.py - Simple One-Liner

For quick processing with default settings:

```bash
python quick_pdf_process.py document.pdf
# Creates: edited_document.pdf
```

## Quality Tiers

| Tier | Model | VRAM | Quality | Speed |
|------|-------|------|---------|-------|
| Fast | BART | 3GB | 7.1/10 | 100% |
| **Balanced** | Mistral-7B | 8GB | 8.7/10 | 50% |
| High | Llama-13B | 16GB | 9.2/10 | 40% |

## Python API

```python
from pdf_processor import PDFProcessor
from pdf_redactor import PDFRedactor

# Process a PDF
processor = PDFProcessor(model_size="small")
processor.process_pdf("input.pdf", "output.pdf", operation="summarize")

# Redact sensitive data
redactor = PDFRedactor()
stats = redactor.redact_pdf("input.pdf", "redacted.pdf")
redactor.save_mappings("mappings.json")
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_pdf_redactor.py -v
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
ruff format .

# Type checking
mypy pdf_processor.py pdf_redactor.py
```

## Project Structure

```text
pdf-processing/
├── pdf_processor.py      # Full PDF processing pipeline
├── pdf_redactor.py       # Data redaction tool
├── quick_pdf_process.py  # Simple one-liner script
├── final_transform.py    # Extensible text transformation framework
├── pdf_utils.py          # Shared utilities
├── constants.py          # Configuration constants
├── tests/                # Test suite (50+ tests)
│   ├── conftest.py       # Shared fixtures
│   └── test_*.py         # Test modules
├── requirements.txt      # Core dependencies
├── requirements-test.txt # Test dependencies
├── pytest.ini            # Pytest configuration
└── .github/workflows/    # CI/CD pipelines
```

## Requirements

- Python 3.11+
- 8GB+ RAM (16GB recommended for large models)
- GPU with 8GB+ VRAM (optional, CPU fallback available)
- poppler-utils (for PDF rasterization)

## Security

- All processing runs locally - documents never leave your machine
- Mapping files are created with restrictive permissions (owner read/write only)
- Post-redaction verification ensures sensitive data is removed
- Metadata and annotations are automatically redacted

**Warning:** Mapping files contain original sensitive data. Store securely and delete when no longer needed.

## Documentation

- [PDF Processing Guide](PDF_PROCESSING_README.md) - Detailed usage guide
- [Quick Reference](QUICK_REFERENCE.md) - Command cheat sheet
- [Model Comparison](MODEL_COMPARISON.md) - Detailed benchmarks
- [Contributing](CONTRIBUTING.md) - How to contribute

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for ML models
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF manipulation
- [Microsoft TrOCR](https://huggingface.co/microsoft/trocr-large-printed) for OCR
- [Mistral AI](https://mistral.ai/) and [Meta](https://ai.meta.com/) for text models
