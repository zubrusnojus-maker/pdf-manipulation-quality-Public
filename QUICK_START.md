# PDF Toolkit Quick Start

> **One-page reference for getting started with PDF Toolkit v1.0+**

## Installation

```bash
# Clone and install
git clone <repository-url>
cd pdf-manipulation-quality-Public
pip install -e .

# System requirement
brew install poppler  # macOS
```

## Usage Patterns

### 1️⃣ Command-Line (Recommended)

```bash
# Process PDF with OCR + text manipulation
pdf-process input.pdf -o output.pdf --use-layout

# Redact PII from PDF
pdf-redact sensitive.pdf -o redacted.pdf --mapping-output mappings.json
```

### 2️⃣ Python Package

```python
# Modern imports (v1.0+)
from pdf_toolkit.core.processor import PDFProcessor
from pdf_toolkit.redaction.redactor import PDFRedactor

# Process PDF
processor = PDFProcessor(model_size="small", use_layout=True)
processor.process_pdf("input.pdf", "output.pdf")

# Redact PII
redactor = PDFRedactor()
redactor.redact_pdf("sensitive.pdf", "redacted.pdf")
```

### 3️⃣ Legacy Scripts (Deprecated)

```bash
# ⚠️ Will be removed in v2.0.0
python pdf_processor.py input.pdf -o output.pdf
python pdf_redactor.py sensitive.pdf -o redacted.pdf
```

## Migration from v0.x

**Old (v0.x):**

```python
from pdf_processor import PDFProcessor
```

**New (v1.0+):**

```python
from pdf_toolkit.core.processor import PDFProcessor
```

See [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md) for complete migration instructions.

## Common Tasks

### Process with High Quality

```bash
pdf-process document.pdf -o result.pdf -m large --use-layout --optimize
```

### Detect PII Only (No Redaction)

```python
from pdf_toolkit.redaction.redactor import PDFRedactor

redactor = PDFRedactor()
text = "Account: 12345678, Balance: $1,234.56"
detections = redactor.detect_sensitive_data(text)
# Returns: [('12345678', 'account_number', 'Account_1'), ...]
```

### Custom Model Configuration

```python
from pdf_toolkit.core.constants import get_model_with_revision

# Get pinned model version
model_config = get_model_with_revision("ocr")
# Returns: {"model": "microsoft/trocr-large-printed", "revision": "c3afae..."}
```

## Quality Tiers

| Tier | Model | VRAM | Quality | Speed | Best For |
|------|-------|------|---------|-------|----------|
| **Fast** | Mistral-7B (small) | 3GB | 7.1/10 | 100% | Quick drafts |
| **Balanced** ⭐ | Mistral-7B (small) | 8GB | 8.7/10 | 50% | Most documents |
| **High** | Llama-2-13B (large) | 16GB | 9.2/10 | 40% | Critical docs |

```bash
# Use balanced (default)
pdf-process doc.pdf -o out.pdf

# Use high quality
pdf-process doc.pdf -o out.pdf -m large
```

## Help & Documentation

- 📖 **Full Guide**: [`README.md`](README.md)
- 🔄 **Migration**: [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md)
- 📝 **Changelog**: [`CHANGELOG.md`](CHANGELOG.md)
- 🎯 **Models**: [`docs/MODEL_VERSIONING.md`](docs/MODEL_VERSIONING.md)
- 🧪 **Testing**: [`docs/TEST_RESULTS.md`](docs/TEST_RESULTS.md)

```bash
# CLI help
pdf-process --help
pdf-redact --help

# Python help
python -c "from pdf_toolkit.core.processor import PDFProcessor; help(PDFProcessor)"
```

## Troubleshooting

### ❌ "No module named 'pdf_toolkit'"

**Solution**: Install package with `pip install -e .`

### ❌ "pdf2image.exceptions.PDFInfoNotInstalledError"

**Solution**: Install poppler-utils

- macOS: `brew install poppler`
- Ubuntu: `sudo apt-get install poppler-utils`

### ❌ DeprecationWarning when importing

**Solution**: Update imports to use `pdf_toolkit` package (see Migration section)

### ❌ "CUDA out of memory"

**Solution**: Use smaller model with `-m small` or reduce DPI with `--dpi 200`

## Next Steps

1. ✅ Run `pip install -e .` to install package
2. ✅ Test CLI: `pdf-process --help`
3. ✅ Try sample: `pdf-process samples/input/test.pdf -o output.pdf`
4. 📖 Read full documentation in [`README.md`](README.md)
5. 🔄 Migrate existing code using [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md)

---

**Version**: 1.0.0 | **License**: MIT | **Python**: 3.10+
