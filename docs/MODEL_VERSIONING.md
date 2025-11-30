# Model Versioning Guide

## Overview

This project uses version pinning for all HuggingFace models to ensure reproducibility across different environments and runs. Model versions are specified using git commit hashes in the `MODEL_REVISIONS` dictionary.

## Current Model Versions

| Model Type | Model Name | Revision | Last Updated |
|------------|-----------|----------|--------------|
| OCR | `microsoft/trocr-large-printed` | `c3afae9716f25251a833d4bd3c6a73c61cdb3d63` | May 2024 |
| Layout | `microsoft/dit-base-finetuned-rvlcdip` | `4b27643ff7e3e9a0e5e62cea8eb77bb5c9a76f93` | Feb 2023 |
| Text (Small) | `mistralai/Mistral-7B-Instruct-v0.2` | `b70aa86578567ba3301b21c8a27bea4e8f6d6d61` | Jul 2025 |
| Text (Large) | `meta-llama/Llama-2-13b-chat-hf` | `0ba94ac9b9e1d5a0037780667e8b219adde1908c` | Apr 2024 |

## How It Works

### Constants Structure

In `src/pdf_toolkit/core/constants.py`:

```python
# Model names
MODELS = {
    "ocr": "microsoft/trocr-large-printed",
    "layout": "microsoft/dit-base-finetuned-rvlcdip",
    "text_small": "mistralai/Mistral-7B-Instruct-v0.2",
    "text_large": "meta-llama/Llama-2-13b-chat-hf",
}

# Version pins (commit hashes)
MODEL_REVISIONS = {
    "ocr": "c3afae9716f25251a833d4bd3c6a73c61cdb3d63",
    "layout": "4b27643ff7e3e9a0e5e62cea8eb77bb5c9a76f93",
    "text_small": "b70aa86578567ba3301b21c8a27bea4e8f6d6d61",
    "text_large": "0ba94ac9b9e1d5a0037780667e8b219adde1908c",
}

# Helper function
def get_model_with_revision(model_key: str) -> dict:
    """Returns {'model': model_name, 'revision': commit_hash}"""
    return {
        "model": MODELS[model_key],
        "revision": MODEL_REVISIONS[model_key],
    }
```

### Usage in Code

Models are loaded using the helper function:

```python
from pdf_toolkit.core.constants import get_model_with_revision

# Load with pinned version
pipeline("image-to-text", **get_model_with_revision("ocr"))
```

This expands to:

```python
pipeline(
    "image-to-text",
    model="microsoft/trocr-large-printed",
    revision="c3afae9716f25251a833d4bd3c6a73c61cdb3d63"
)
```

## Updating Model Versions

### When to Update

- Security fixes in model dependencies
- Performance improvements in newer versions
- Bug fixes in model implementations
- Major version releases with breaking changes

### How to Update

1. **Find the model on HuggingFace Hub:**

   ```
   https://huggingface.co/{model_name}
   ```

2. **Navigate to the "Files and versions" tab** to see all commits

3. **Identify a stable commit:**
   - Look for tagged releases (preferred)
   - Or use recent commits from the main branch
   - Verify the commit date and any release notes

4. **Test the new version:**

   ```python
   # Temporarily test new revision
   pipeline("image-to-text", 
            model="microsoft/trocr-large-printed",
            revision="new_commit_hash_here")
   ```

5. **Update `MODEL_REVISIONS` in constants.py:**

   ```python
   MODEL_REVISIONS = {
       "ocr": "new_commit_hash_here",  # Update with date comment
       # ... other models
   }
   ```

6. **Run tests to verify compatibility:**

   ```bash
   pytest tests/
   ```

7. **Document the change in CHANGELOG.md**

### Finding Commit Hashes

Using HuggingFace Hub:

```python
from huggingface_hub import list_repo_commits

commits = list_repo_commits("microsoft/trocr-large-printed", repo_type="model")
for commit in commits[:5]:
    print(f"{commit.commit_id}: {commit.title}")
```

Or visit the model page directly:

```
https://huggingface.co/microsoft/trocr-large-printed/commits/main
```

## Backward Compatibility

The `MODELS` dictionary is maintained for backward compatibility. Any existing code that directly accesses `MODELS["ocr"]` will continue to work, but won't benefit from version pinning.

To use version pinning, update code to use `get_model_with_revision()`:

```python
# Old (unpinned)
pipeline("image-to-text", model=MODELS["ocr"])

# New (pinned)
pipeline("image-to-text", **get_model_with_revision("ocr"))
```

## Benefits of Version Pinning

1. **Reproducibility**: Same results across different environments and time periods
2. **Stability**: Avoid unexpected changes from model updates
3. **Debugging**: Easier to track down issues to specific model versions
4. **Compliance**: Meet requirements for production systems that need version tracking
5. **Testing**: Ensure tests run against the same model versions

## Notes

- **Llama-2-13b-chat-hf** requires gated access. Users must accept the license on HuggingFace Hub
- Commit hashes are permanent and immutable in git
- Tags (like "v1.0") are also supported but less common in HuggingFace repos
- Models are cached locally after first download in `~/.cache/huggingface/`
