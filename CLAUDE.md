# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quality-focused PDF processing toolkit combining OCR (TrOCR Large), layout analysis (DiT), and AI text manipulation (Llama-2/Mistral) to extract, summarize/rewrite, and re-insert text into PDFs. All models run 100% locally after initial download.

## Commands

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html -v

# Run specific test file
pytest tests/test_pdf_redactor.py -v
```

### PDF Processing
```bash
# Quick processing (balanced quality, 8GB VRAM)
python quick_pdf_process.py document.pdf

# Full-featured with options
python pdf_processor.py input.pdf -o output.pdf --model-size large --use-layout --dpi 600

# PDF redaction/anonymization
python pdf_redactor.py input.pdf -o output.pdf --mapping-output mappings.json
```

### Dependencies
```bash
pip install -r requirements.txt       # Core dependencies
pip install -r requirements-test.txt  # Test dependencies (pytest, requests, numpy)

# Or install as a package
pip install -e .
```

System requirement: `poppler-utils` for pdf2image rasterization.

### Installed CLI Commands
After `pip install -e .`, these commands are available:
```bash
pdf-process input.pdf -o output.pdf   # Process PDF with OCR
pdf-redact input.pdf -o output.pdf    # Anonymize PDF data
```

## Architecture

### Package Structure

```
src/pdf_toolkit/
├── __init__.py           # Package exports: PDFProcessor, PDFRedactor, LayoutRegion
├── core/
│   ├── __init__.py
│   ├── processor.py      # PDFProcessor class - main processing logic
│   ├── layout.py         # LayoutRegion, LAYOUT_TYPES, DOCUMENT_TYPE_MAPPING
│   └── constants.py      # MIN_TEXT_LENGTH, MODELS, MARGIN_POINTS
├── models/
│   ├── __init__.py
│   └── loader.py         # ModelLoader class for loading TrOCR, DiT, Mistral/Llama
├── redaction/
│   ├── __init__.py
│   ├── redactor.py       # PDFRedactor class for data anonymization
│   └── patterns.py       # REDACTION_PATTERNS for sensitive data detection
├── utils/
│   ├── __init__.py
│   └── pdf_utils.py      # rasterize_pdf(), optimize_pdf()
└── cli/
    ├── __init__.py
    ├── processor_cli.py  # CLI entry point for pdf-process
    └── redactor_cli.py   # CLI entry point for pdf-redact
```

**Backwards compatibility**: The root-level `pdf_processor.py` and `pdf_redactor.py` files are thin wrappers that re-export from the package, so existing scripts continue to work.

### Processing Pipeline

1. **Rasterization** - PDF to images via `pdf2image` (300 DPI default)
2. **OCR** - `microsoft/trocr-large-printed` extracts text from images
3. **Layout Analysis** (optional) - `microsoft/dit-base-finetuned-rvlcdip` classifies document type
4. **Text Manipulation** - Mistral-7B (balanced) or Llama-2-13B (high quality) for summarization/rewriting
5. **Re-insertion** - PyMuPDF (`fitz`) overlays modified text with layout-aware formatting
6. **Optimization** (optional) - `pikepdf` for file size reduction

### Layout Analysis

When `--use-layout` is enabled, the processor:
- Classifies document type (invoice, letter, form, presentation, etc.)
- Adds document context to LLM prompts for better summarization
- Adjusts text formatting (font size, alignment) based on document type

| Document Type | Layout Region | Font Multiplier | Alignment |
|--------------|---------------|-----------------|-----------|
| presentation, advertisement | header | 1.2x | centered |
| letter, email, memo | paragraph | 1.0x | left |
| invoice, form, budget | table | 0.9x | left |
| footer | footer | 0.8x | centered |

### Key Files

| File | Purpose |
|------|---------|
| `src/pdf_toolkit/core/processor.py` | PDFProcessor class with OCR and text manipulation |
| `src/pdf_toolkit/redaction/redactor.py` | PDFRedactor for data anonymization |
| `src/pdf_toolkit/models/loader.py` | ModelLoader for managing ML models |
| `pdf_processor.py` | Backwards-compatible wrapper with CLI |
| `pdf_redactor.py` | Backwards-compatible wrapper with CLI |
| `quick_pdf_process.py` | Single-script one-liner for simple use cases |

### Quality Tiers

| Tier | Text Model | VRAM | Quality | Use Case |
|------|-----------|------|---------|----------|
| Fast | BART-Large | 3GB | 7.1/10 | High-volume, non-critical |
| Balanced (default) | Mistral-7B | 8GB | 8.7/10 | General business documents |
| High | Llama-2-13B | 16GB | 9.2/10 | Legal, medical, research |

## Code Conventions

### Device Configuration
```python
# Always auto-detect GPU
device = 0 if torch.cuda.is_available() else -1

# For large models use device_map
pipeline("text-generation", model="...", device_map="auto", torch_dtype=torch.float16)
```

### Instruction Model Prompting
```python
# Mistral/Llama require this exact format
prompt = f"<s>[INST] {instruction}\n\n{text}\n\n[/INST]"
output = result[0]["generated_text"].split("[/INST]")[-1].strip()
```

### Text Handling
```python
# Truncate before processing (Mistral: 1024, Llama: 2048)
max_length = 2048 if model_size == "large" else 1024
text = text[:max_length] if len(text) > max_length else text

# Skip very short text (< 50 chars)
if len(text.strip()) < 50:
    continue
```

### PyMuPDF Text Insertion
```python
# Standard: 72pt margins (1 inch), black text, left-aligned
page.insert_textbox(
    fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72),
    new_text,
    fontsize=12,
    color=(0, 0, 0),
    align=0
)
```

## Model Downloads

First run downloads models to `~/.cache/huggingface/hub/`:
- TrOCR Large: ~2.2GB
- DiT (layout): ~350MB
- Mistral-7B: ~14GB
- Llama-2-13B: ~26GB

Verify downloads completed before testing - incomplete downloads cause cryptic errors.

## Task Master AI Integration

This project uses Task Master AI for task management. See `.rules` for complete documentation.

### Essential Commands
```bash
task-master list                              # Show all tasks
task-master next                              # Get next task to work on
task-master show <id>                         # View task details (e.g., 1.2)
task-master set-status --id=<id> --status=done  # Mark complete

# Task creation and expansion
task-master add-task --prompt="description" --research
task-master expand --id=<id> --research --force
task-master analyze-complexity --research
```

### Key Files
- `.taskmaster/tasks/tasks.json` - Main task database (auto-managed, don't edit manually)
- `.taskmaster/config.json` - AI model config (use `task-master models` to modify)
- `.taskmaster/docs/prd.txt` - Product requirements document
- `.env` - API keys (copy from `.env.example`)

## Common Pitfalls

- Don't use generic summarization models (BART) for quality-critical work
- Don't forget instruction formatting for Mistral/Llama (`<s>[INST]...[/INST]`)
- Don't hardcode `device=0` - always check `torch.cuda.is_available()`
- Don't skip text truncation - long text exceeds model context
- Always wrap model calls in try/except with fallback to original text
- Don't manually edit `tasks.json` or `.taskmaster/config.json` - use Task Master commands
