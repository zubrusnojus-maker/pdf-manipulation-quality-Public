# Migration Guide

This guide helps you transition from legacy import patterns to the modern package structure introduced in v1.0.

## Overview

The PDF Toolkit has been restructured from monolithic scripts to a well-organized Python package (`pdf_toolkit`). This migration enables:

- ✅ Better code organization and maintainability
- ✅ Proper Python package installation with `pip install -e .`
- ✅ Command-line tools accessible from anywhere
- ✅ Cleaner imports and better IDE support
- ✅ MCP (Model Context Protocol) server integration

## Quick Migration Checklist

- [ ] Install the package: `pip install -e .`
- [ ] Update imports from legacy to package imports
- [ ] Replace direct script calls with CLI commands
- [ ] Test your code with the new structure
- [ ] Update documentation/scripts to use new patterns

## Installation

### New Installation Method (v1.0+)

```bash
# Clone the repository
git clone <repository-url>
cd pdf-manipulation-quality-Public

# Install as a package (editable mode for development)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

After installation, CLI commands are available system-wide:

```bash
pdf-process input.pdf -o output.pdf
pdf-redact sensitive.pdf -o redacted.pdf
```

### System Requirements

- Python 3.10 or higher
- `poppler-utils` for PDF rasterization:
  - **macOS**: `brew install poppler`
  - **Ubuntu/Debian**: `sudo apt-get install poppler-utils`
  - **Windows**: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows)

## Import Pattern Migration

### PDFProcessor Class

#### Legacy (v0.x)

```python
# Old way - importing from root-level script
from pdf_processor import PDFProcessor

processor = PDFProcessor()
processor.process_pdf("input.pdf", "output.pdf")
```

#### Modern (v1.0+)

```python
# New way - importing from package
from pdf_toolkit import PDFProcessor

processor = PDFProcessor()
processor.process_pdf("input.pdf", "output.pdf")
```

### PDFRedactor Class

#### Legacy (v0.x)

```python
# Old way
from pdf_redactor import PDFRedactor

redactor = PDFRedactor()
stats = redactor.redact_pdf("input.pdf", "output.pdf")
```

#### Modern (v1.0+)

```python
# New way
from pdf_toolkit import PDFRedactor

redactor = PDFRedactor()
stats = redactor.redact_pdf("input.pdf", "output.pdf")
```

### Layout Analysis

#### Legacy (v0.x)

```python
# Old way - not exported
# Had to dig into implementation details
```

#### Modern (v1.0+)

```python
# New way - clean exports
from pdf_toolkit import LayoutRegion
from pdf_toolkit.core.layout import LAYOUT_TYPES, DOCUMENT_TYPE_MAPPING

# Use layout regions in your code
region = LayoutRegion(name="header", confidence=0.95)
```

### Model Loading

#### Legacy (v0.x)

```python
# Old way - manual model management
from transformers import pipeline

ocr = pipeline("image-to-text", model="microsoft/trocr-large-printed")
```

#### Modern (v1.0+)

```python
# New way - centralized model loader
from pdf_toolkit.models import ModelLoader

loader = ModelLoader(model_size="small")
ocr_pipeline = loader.get_ocr_pipeline()
text_pipeline = loader.get_text_pipeline()
```

### Utility Functions

#### Legacy (v0.x)

```python
# Old way - not exported/accessible
# Functions buried in scripts
```

#### Modern (v1.0+)

```python
# New way - clean utility exports
from pdf_toolkit.utils import rasterize_pdf, optimize_pdf

# Rasterize PDF to images
images = rasterize_pdf("document.pdf", dpi=300)

# Optimize file size
optimize_pdf("input.pdf", "output.pdf")
```

## CLI Command Migration

### PDF Processing

#### Legacy (v0.x)

```bash
# Old way - direct script execution
python pdf_processor.py input.pdf -o output.pdf --model-size large

# Or quick processing script
python quick_pdf_process.py document.pdf
```

#### Modern (v1.0+)

```bash
# New way - installed CLI command
pdf-process input.pdf -o output.pdf --model-size large

# Same functionality, cleaner invocation
pdf-process document.pdf
```

**Available options:**

```bash
pdf-process --help

Options:
  -o, --output PATH          Output PDF path (default: edited_<input>.pdf)
  --operation [summarize|rewrite]  Operation to perform (default: summarize)
  --model-size [small|large]  Model size: small=Mistral-7B, large=Llama-2-13B
  --use-layout               Enable layout analysis
  --dpi INTEGER              DPI for rasterization (default: 300)
  --optimize                 Optimize output with pikepdf
```

### PDF Redaction

#### Legacy (v0.x)

```bash
# Old way
python pdf_redactor.py input.pdf -o output.pdf --mapping-output mappings.json
```

#### Modern (v1.0+)

```bash
# New way
pdf-redact input.pdf -o output.pdf --mapping-output mappings.json
```

**Available options:**

```bash
pdf-redact --help

Options:
  -o, --output PATH          Output PDF path (default: redacted_<input>.pdf)
  --mapping-output PATH      Save mappings to JSON file
  --preserve-dates           Don't redact dates
```

## Python API Migration

### Basic Processing

#### Legacy (v0.x)

```python
import sys
sys.path.append('/path/to/scripts')  # Had to manually manage paths

from pdf_processor import PDFProcessor

processor = PDFProcessor(model_size="small")
processor.process_pdf("input.pdf", "output.pdf")
```

#### Modern (v1.0+)

```python
from pdf_toolkit import PDFProcessor

processor = PDFProcessor(model_size="small")
processor.process_pdf("input.pdf", "output.pdf")
```

### Advanced Configuration

#### Legacy (v0.x)

```python
# Old way - limited configuration options
from pdf_processor import PDFProcessor

processor = PDFProcessor(model_size="large")
# Had to modify source code for custom settings
```

#### Modern (v1.0+)

```python
# New way - flexible configuration
from pdf_toolkit import PDFProcessor

processor = PDFProcessor(
    model_size="large",
    use_layout=True,
)

# Process with custom settings
processor.process_pdf(
    input_path="document.pdf",
    output_path="processed.pdf",
    operation="rewrite",
    dpi=600,
    optimize=True
)
```

### PII Detection and Redaction

#### Legacy (v0.x)

```python
from pdf_redactor import PDFRedactor

redactor = PDFRedactor()
redactor.redact_pdf("input.pdf", "output.pdf")
redactor.save_mappings("mappings.json")
```

#### Modern (v1.0+)

```python
from pdf_toolkit import PDFRedactor

redactor = PDFRedactor()

# Detect PII without modifying text
detections = redactor.detect_sensitive_data(
    "Invoice from John Smith for $1,234.56"
)

# Redact PDF
stats = redactor.redact_pdf(
    "input.pdf",
    "output.pdf",
    preserve_dates=False
)

# Save mappings for later reference
redactor.save_mappings("mappings.json")
```

## MCP Server Integration (New in v1.0)

The package now includes MCP (Model Context Protocol) server support for integration with AI assistants like Claude.

### Starting the MCP Server

```bash
# Using the mcp_server module
python -m pdf_toolkit.mcp_server
```

### MCP Client Configuration

Add to your MCP client config (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "pdf-toolkit": {
      "command": "python",
      "args": ["-m", "pdf_toolkit.mcp_server"]
    }
  }
}
```

### Available MCP Tools

- `redact_pdf` - Redact sensitive information from PDFs
- `detect_pii` - Detect PII in text
- `process_pdf` - OCR and text manipulation
- `optimize_pdf_file` - Optimize PDF file size

## HuggingFace Space Deployment (New in v1.0)

The toolkit can be deployed as a Gradio app with MCP integration:

```bash
# Run locally
python app.py

# Deploy to HuggingFace Spaces
# 1. Create a new Space on HuggingFace
# 2. Push the repository
# 3. Ensure requirements-hf.txt is used
```

### Accessing the Deployed MCP Server

```json
{
  "mcpServers": {
    "pdf-toolkit": {
      "url": "https://YOUR-SPACE-URL/gradio_api/mcp/sse"
    }
  }
}
```

## Package Structure Reference

### Before (v0.x)

```
pdf-manipulation-quality-Public/
├── pdf_processor.py          # Monolithic script
├── pdf_redactor.py           # Monolithic script
├── quick_pdf_process.py      # Simple script
└── requirements.txt
```

### After (v1.0+)

```
pdf-manipulation-quality-Public/
├── src/pdf_toolkit/          # Main package
│   ├── __init__.py          # Public API exports
│   ├── core/                # Core processing logic
│   │   ├── processor.py     # PDFProcessor class
│   │   ├── layout.py        # Layout analysis
│   │   └── constants.py     # Configuration
│   ├── models/              # Model management
│   │   └── loader.py        # ModelLoader class
│   ├── redaction/           # PII detection/redaction
│   │   ├── redactor.py      # PDFRedactor class
│   │   └── patterns.py      # Regex patterns
│   ├── utils/               # Utility functions
│   │   └── pdf_utils.py     # PDF operations
│   ├── cli/                 # Command-line interfaces
│   │   ├── processor_cli.py # pdf-process command
│   │   └── redactor_cli.py  # pdf-redact command
│   └── mcp_server.py        # MCP server
├── pdf_processor.py          # Legacy wrapper (deprecated)
├── pdf_redactor.py           # Legacy wrapper (deprecated)
├── app.py                    # Gradio/HF Space app
└── pyproject.toml            # Package metadata
```

## Deprecation Timeline

### v1.0 (Current)

- ✅ New package structure fully functional
- ✅ Legacy wrappers (`pdf_processor.py`, `pdf_redactor.py`) still work
- ⚠️ Direct script imports are **deprecated** but supported
- ✅ CLI commands available after `pip install -e .`

### v1.1 (Planned)

- ⚠️ Legacy wrappers show deprecation warnings
- ⚠️ Documentation focuses on package imports
- ✅ All examples use modern imports

### v2.0 (Future)

- ❌ Legacy wrappers removed
- ❌ Direct script imports no longer supported
- ✅ Package-only imports required

## Troubleshooting

### Import Errors

**Problem:** `ImportError: No module named 'pdf_toolkit'`

**Solution:**

```bash
# Install the package
pip install -e .

# Or add to PYTHONPATH temporarily
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Command Not Found

**Problem:** `pdf-process: command not found`

**Solution:**

```bash
# Reinstall with pip
pip install -e .

# Check if installed
pip list | grep pdf-toolkit

# Use with python -m if not in PATH
python -m pdf_toolkit.cli.processor_cli input.pdf -o output.pdf
```

### Legacy Scripts Not Working

**Problem:** Old scripts using `from pdf_processor import PDFProcessor` fail

**Solution:** Legacy wrappers should work, but if they don't:

```bash
# Option 1: Update imports (recommended)
# Change: from pdf_processor import PDFProcessor
# To:     from pdf_toolkit import PDFProcessor

# Option 2: Use legacy wrappers (temporary)
python pdf_processor.py input.pdf -o output.pdf

# Option 3: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python your_script.py
```

### Model Download Issues

**Problem:** Models fail to download or load

**Solution:**

```bash
# Ensure transformers is up to date
pip install --upgrade transformers torch

# Check disk space (~30GB needed for large models)
df -h

# Manually download to cache
python -c "from transformers import pipeline; pipeline('image-to-text', model='microsoft/trocr-large-printed')"
```

### Permission Issues on CLI Commands

**Problem:** CLI commands not executable after install

**Solution:**

```bash
# Reinstall in user mode
pip install --user -e .

# Or use python -m
python -m pdf_toolkit.cli.processor_cli --help
```

## Testing Your Migration

After migrating, run these tests to ensure everything works:

```bash
# 1. Test package import
python -c "from pdf_toolkit import PDFProcessor, PDFRedactor; print('✓ Imports work')"

# 2. Test CLI commands
pdf-process --help
pdf-redact --help

# 3. Run test suite
pytest tests/ -v

# 4. Process a sample PDF
pdf-process samples/input/test.pdf -o samples/output/test_out.pdf
```

## Getting Help

- 📖 **Full Documentation**: See `docs/PDF_PROCESSING_README.md`
- 🚀 **Quick Reference**: See `docs/QUICK_REFERENCE.md`
- 📊 **Model Comparison**: See `docs/MODEL_COMPARISON.md`
- 🔧 **Code Reference**: See `CLAUDE.md` for implementation details
- 🐛 **Issues**: Check GitHub issues or create a new one

## Summary

The migration to v1.0 package structure provides:

1. **Better organization** - Logical module separation
2. **Easier installation** - Standard `pip install`
3. **CLI convenience** - System-wide commands
4. **Better testing** - Proper package structure
5. **MCP integration** - AI assistant compatibility
6. **HuggingFace deployment** - Web UI and API

**Action items:**

1. Run `pip install -e .`
2. Update imports: `from pdf_toolkit import PDFProcessor`
3. Replace script calls with CLI: `pdf-process` instead of `python pdf_processor.py`
4. Test your code with the new structure
5. Remove manual path manipulations

---

**Last Updated:** November 30, 2025
