# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quality-focused PDF processing toolkit combining OCR (TrOCR Large), layout analysis (LayoutLMv3), and AI text manipulation (Llama-2/Mistral) to extract, summarize/rewrite, and re-insert text into PDFs. All models run 100% locally after initial download.

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
```

System requirement: `poppler-utils` for pdf2image rasterization.

## Architecture

### Processing Pipeline

1. **Rasterization** - PDF to images via `pdf2image` (300 DPI default)
2. **OCR** - `microsoft/trocr-large-printed` extracts text from images
3. **Text Manipulation** - Mistral-7B (balanced) or Llama-2-13B (high quality) for summarization/rewriting
4. **Re-insertion** - PyMuPDF (`fitz`) overlays modified text onto original pages
5. **Optimization** (optional) - `pikepdf` for file size reduction

### Key Files

| File | Purpose |
|------|---------|
| `pdf_processor.py` | Main class-based implementation with CLI |
| `quick_pdf_process.py` | Single-script one-liner for simple use cases |
| `pdf_redactor.py` | PDF data anonymization/redaction tool |

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
- Mistral-7B: ~14GB
- Llama-2-13B: ~26GB

Verify downloads completed before testing - incomplete downloads cause cryptic errors.

## Common Pitfalls

- Don't use generic summarization models (BART) for quality-critical work
- Don't forget instruction formatting for Mistral/Llama (`<s>[INST]...[/INST]`)
- Don't hardcode `device=0` - always check `torch.cuda.is_available()`
- Don't skip text truncation - long text exceeds model context
- Always wrap model calls in try/except with fallback to original text
