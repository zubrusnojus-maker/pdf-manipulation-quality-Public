# PDF Processing with OCR and AI Text Manipulation

## Project Overview

Quality-focused PDF processing toolkit combining OCR (TrOCR Large), layout analysis (LayoutLMv3), and AI text manipulation (Llama-2/Mistral) to extract, summarize/rewrite, and re-insert text into PDFs. Optimized for November 2025 state-of-the-art models.

## Architecture & Design Philosophy

### Quality-First Approach

- **Primary Goal**: Maximum accuracy over speed. Uses best-in-class models (TrOCR Large: 97% accuracy)
- **Three Quality Tiers**: Fast (3GB VRAM), Balanced (8GB, **recommended**), High (16GB)
- **Model Selection**: Deliberately chose Mistral-7B-Instruct as default (balances quality/resources)

### Processing Pipeline

1. **Rasterization** → PDF to images via `pdf2image` (300 DPI default)
2. **OCR** → `microsoft/trocr-large-printed` extracts text from images
3. **Text Manipulation** → Llama-2-13B (high) or Mistral-7B-Instruct (balanced) for summarization/rewriting
4. **Re-insertion** → PyMuPDF (`fitz`) overlays modified text onto original pages
5. **Optional Optimization** → `pikepdf` for file size reduction

### Key Files

- `pdf_processor.py` - Main class-based implementation with full CLI
- `quick_pdf_process.py` - Single-script one-liner for simple use cases
- `requirements-pdf.txt` - PDF-specific dependencies (models, libraries)
- `requirements.txt` - Base data science stack (notebooks only)

## Critical Conventions

### Model Usage Patterns

**Always specify device correctly:**

```python
# ✓ Correct: Auto-detect GPU
device=0 if torch.cuda.is_available() else -1

# ✓ Correct: Large models with auto device mapping
pipeline("text-generation", model="meta-llama/Llama-2-13b-chat-hf", 
         device_map="auto", torch_dtype=torch.float16)
```text

**Instruction model prompting (Mistral/Llama):**

```python
# ✓ Follow this exact format for Mistral-7B-Instruct
prompt = f"<s>[INST] {instruction}\n\n{text}\n\n[/INST]"

# Extract response after [/INST] tag
output = result[0]["generated_text"].split("[/INST]")[-1].strip()
```text

### Text Length Handling

**Always truncate before processing:**

```python
# ✓ Prevent model context overflow
max_length = 2048 if model_size == "large" else 1024
if len(text) > max_length:
    text = text[:max_length]
```text

**Skip very short text:**

```python
# ✓ Avoid processing noise
if len(text.strip()) < 50:
    rewritten.append(text)  # Pass through
    continue
```text

### PyMuPDF Text Insertion

**Standard textbox pattern:**

```python
# ✓ 72pt margins (1 inch), black text, left-aligned
page.insert_textbox(
    fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72),
    new_text,
    fontsize=12,
    color=(0, 0, 0),
    align=0
)
```text

## Development Workflows

### Running PDF Processing

**Quick processing (one command):**

```bash
python quick_pdf_process.py document.pdf
# Creates: edited_document.pdf
```text

**Full-featured processing:**

```bash
# Balanced quality (recommended - 8GB VRAM)
python pdf_processor.py input.pdf

# Maximum quality (16GB VRAM)
python pdf_processor.py input.pdf --model-size large --use-layout

# Rewrite instead of summarize
python pdf_processor.py input.pdf --operation rewrite

# High DPI for scanned documents
python pdf_processor.py input.pdf --dpi 600 --optimize
```text

## Testing Strategy

### Testing Philosophy

**No formal test suite exists** - this is a proof-of-concept with manual verification. Test manually before committing changes that affect core pipeline stages.

### Manual Testing Workflow

**1. Basic PDF Operations (No Models)**

```python
import fitz

# Test loading and text extraction
doc = fitz.open('test.pdf')
assert len(doc) > 0, "PDF should have pages"
text = doc[0].get_text()
print(f"Extracted {len(text)} chars from page 1")

# Test text insertion
page = doc[0]
page.insert_textbox(
    fitz.Rect(72, 72, 200, 200),
    "TEST TEXT",
    fontsize=12,
    color=(1, 0, 0)
)
doc.save("test_output.pdf")
doc.close()
```text

**2. OCR Pipeline Test (Requires GPU)**

```python
import pdf2image
from transformers import pipeline
import torch

# Test rasterization
images = pdf2image.convert_from_path('test.pdf', dpi=300)
print(f"Rasterized {len(images)} pages")

# Test OCR model
ocr = pipeline(
    "image-to-text",
    model="microsoft/trocr-large-printed",
    device=0 if torch.cuda.is_available() else -1
)
result = ocr(images[0])
print(f"OCR result: {result[0]['generated_text'][:100]}...")
```text

**3. Text Manipulation Test**

```python
# Test Mistral-7B instruction following
summarizer = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype=torch.float16,
    max_new_tokens=256
)

test_text = "Your sample text here..."
prompt = f"<s>[INST] Summarize this text:\n\n{test_text}\n\n[/INST]"
result = summarizer(prompt)
output = result[0]["generated_text"].split("[/INST]")[-1].strip()
print(f"Summary: {output}")
```text

**4. Full End-to-End Verification**

```bash
# Use small test PDF (1-2 pages)
python pdf_processor.py test.pdf -o output.pdf

# Verify output
# - Check file was created
# - Open in PDF viewer to inspect quality
# - Compare input vs output text content
```text

### Critical Test Cases

**Edge cases to verify manually:**

```python
# Empty/very short text (should pass through unchanged)
assert len("short".strip()) < 50  # Skip processing

# Long text (should truncate before model)
long_text = "x" * 5000
truncated = long_text[:1024]  # Mistral limit
assert len(truncated) == 1024

# Multi-page documents
doc = fitz.open('multi_page.pdf')
assert len(images) == len(doc)  # All pages processed
```text

**Error handling patterns:**

```python
# ✓ Graceful degradation on model failure
try:
    result = summarizer(prompt)
    output = extract_response(result)
except Exception as e:
    print(f"Warning: Failed to process text: {e}")
    output = original_text  # Fall back to original
```text

### Performance Testing

**Benchmark on known documents:**

```bash
# Small doc (1-5 pages): ~1-2 min
time python pdf_processor.py small.pdf

# Medium doc (20-50 pages): ~10-15 min
time python pdf_processor.py medium.pdf

# Large doc (500 pages): ~28 min (Mistral) / ~45 min (Llama)
time python pdf_processor.py --model-size large large.pdf
```text

**Memory monitoring:**

```bash
# Watch GPU memory during processing
watch -n 1 nvidia-smi

# Expected VRAM usage:
# - Balanced: ~8GB peak
# - High: ~16GB peak
# - Fast: ~3GB peak
```text

### Regression Testing Checklist

When modifying code, manually verify:

- [ ] PDF loads without errors
- [ ] All pages are rasterized (count matches)
- [ ] OCR extracts readable text (spot check first page)
- [ ] Text manipulation produces coherent output (not truncated/garbled)
- [ ] Output PDF opens in reader (not corrupted)
- [ ] File size is reasonable (not 10x larger)
- [ ] GPU memory doesn't leak across pages (monitor nvidia-smi)
- [ ] CPU fallback works (set `CUDA_VISIBLE_DEVICES=""`)

### Model Download Verification

**First run downloads large models:**

```bash
# Mistral-7B: ~14GB (cached in ~/.cache/huggingface/)
# Llama-2-13B: ~26GB
# TrOCR Large: ~2.2GB

# Check cache location
ls -lh ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2

# Verify download completed before testing
# Incomplete downloads cause cryptic errors
```text

### Known Testing Limitations

- **No unit tests** - Add pytest framework if productionizing
- **No CI/CD** - Manual verification required
- **No mocking** - Tests require actual models (slow, GPU-dependent)
- **No coverage tracking** - Critical paths untested
- **Error cases untested** - Corrupted PDFs, unsupported formats, out-of-memory

### Notebook Usage

Three sample notebooks demonstrate data science workflows (separate from PDF processing):

- `notebooks/image-classifier.ipynb` - PyTorch image classification
- `notebooks/matplotlib.ipynb` - Data visualization examples
- `notebooks/population.ipynb` - CSV data analysis

Use `requirements.txt` for notebooks, `requirements-pdf.txt` for PDF scripts.

## External Dependencies & Integration

### GPU Requirements

- **Balanced (default)**: 8GB VRAM (RTX 3090, RTX 4070 Ti, A5000)
- **High quality**: 16GB VRAM (A100, RTX 4090, A6000)
- **CPU fallback**: Works but very slow (~0.6 tokens/sec for Mistral)

### Key Libraries

- **PyMuPDF (`fitz`)**: PDF manipulation, text insertion
- **pdf2image**: Rasterization (requires `poppler-utils` system package)
- **transformers**: Hugging Face model loading
- **pikepdf**: Optional PDF optimization/compression

### Model Sources & Local vs Remote

**This project ALREADY uses local models** - all models from Hugging Face Hub run locally on your GPU/CPU:

- `microsoft/trocr-large-printed` (OCR, ~2.2GB)
- `mistralai/Mistral-7B-Instruct-v0.2` (text, ~14GB, **default**)
- `meta-llama/Llama-2-13b-chat-hf` (text, ~26GB, high quality)
- `microsoft/layoutlmv3-base` (layout, ~0.5GB, optional)

**How it works:**

1. First run downloads models to `~/.cache/huggingface/hub/`
2. Models stay cached permanently (no re-download)
3. Inference runs 100% locally with zero API calls
4. No internet required after initial download

**Benefits over API services (OpenAI, Anthropic, etc.):**

- ✅ **Zero recurring costs** - no per-token charges
- ✅ **Complete privacy** - documents never leave your machine
- ✅ **No rate limits** - process unlimited documents
- ✅ **Offline capable** - works without internet (post-download)
- ✅ **Reproducible** - same model version always
- ✅ **No API key management** - no credentials needed

**Trade-offs:**

- ❌ **Initial setup cost** - requires GPU hardware (~$500-2000)
- ❌ **Storage requirements** - 14-26GB per model
- ❌ **Slower cold start** - model loading takes 30-60 seconds
- ❌ **Limited to GPU memory** - can't scale beyond hardware
- ❌ **Self-managed updates** - must manually switch to newer models

### When to Consider API Services Instead

**Use OpenAI/Anthropic APIs if:**

- Processing <100 documents/month (cost-effective at low volume)
- Need GPT-4/Claude quality (better than Llama-2-13B)
- Don't have GPU hardware available
- Require dynamic scaling (thousands of concurrent requests)
- Want latest models without manual updates

**Stick with local models if:**

- Processing hundreds/thousands of documents regularly
- Have privacy/compliance requirements (HIPAA, GDPR)
- Have GPU hardware already (existing ML infrastructure)
- Need predictable costs (no surprise API bills)
- Require deterministic output (same model version)

### Alternative Local Model Options

**For lower resource requirements:**

```python
# Quantized models (4-bit) - reduce VRAM by ~4x
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=quantization_config,
    device_map="auto"
)
# Mistral-7B: 8GB → 2GB VRAM (with minimal quality loss)
```text

**For CPU-only systems:**

```python
# Use GGUF format with llama.cpp (not implemented in this project)
# Or use smaller BART model (already supported)
processor = PDFProcessor(model_size="fast")  # Uses BART (3GB)
```text

**Local model serving alternatives:**

- **Ollama**: Simple local LLM server (easier than raw transformers)
- **vLLM**: High-performance inference server (better throughput)
- **LocalAI**: OpenAI-compatible API with local models
- **LM Studio**: GUI for running local models

**Current implementation uses direct transformers pipeline** - simplest approach, no server needed.

## Common Pitfalls

### ❌ Don't use generic summarization models

```python
# ❌ BART produces low-quality output (7.1/10 vs 8.7/10)
pipeline("summarization", model="facebook/bart-large-cnn")
```text

### ❌ Don't forget instruction formatting

```python
# ❌ Mistral/Llama need proper prompt format
result = summariser(text)  # Wrong - will generate poor output

# ✓ Use instruction template
result = summariser(f"<s>[INST] Summarize: {text} [/INST]")
```text

### ❌ Don't hardcode device without checking

```python
# ❌ Crashes on CPU-only systems
pipeline("image-to-text", model="...", device=0)

# ✓ Check GPU availability
device=0 if torch.cuda.is_available() else -1
```text

### ❌ Don't skip error handling on model calls

```python
# ❌ Model failures crash entire pipeline
result = self.summariser(prompt)
output = result[0]["generated_text"]

# ✓ Graceful fallback to original text
try:
    result = self.summariser(prompt)
    output = result[0].get("generated_text", text)
except Exception as e:
    print(f"Warning: Failed to process: {e}")
    output = text  # Continue with original
```text

### ❌ Don't process without truncating

```python
# ❌ Long text exceeds model context, causes truncation warnings/errors
result = summariser(long_text)

# ✓ Truncate before processing
max_len = 1024
text = text[:max_len] if len(text) > max_len else text
```text

### ❌ Don't assume models are downloaded

```python
# ❌ First run will pause for large downloads without warning
processor.load_models()  # Could take 30+ minutes

# ✓ Inform users or pre-check cache
print("First run downloads ~14GB models (~15-30 min)")
# Check: ~/.cache/huggingface/hub/models--mistralai--*
```text

## Performance Benchmarks

See `MODEL_COMPARISON.md` for detailed metrics. Key takeaways:

- **TrOCR Large**: 97.2% accuracy (3% better than base model)
- **Mistral-7B**: 8.7/10 quality, 2 hallucinations per doc, 5.1 tok/s
- **Llama-2-13B**: 9.2/10 quality, 1 hallucination per doc, 2.5 tok/s
- **Processing time**: ~28min for 500-page doc (Mistral), ~45min (Llama)

## Resource References

- **Setup**: `PDF_PROCESSING_README.md` - Full installation and usage guide
- **Quick Start**: `QUICK_REFERENCE.md` - Command cheat sheet by quality tier
- **Model Selection**: `MODEL_COMPARISON.md` - Detailed benchmarks and rationale
- **Testing**: `TEST_RESULTS.md` - Validation results on real documents
