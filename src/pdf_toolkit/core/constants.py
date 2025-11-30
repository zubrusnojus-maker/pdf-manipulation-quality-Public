"""Constants used throughout the PDF processing pipeline."""

# Text processing
MIN_TEXT_LENGTH = 50
MAX_INPUT_LENGTH_SMALL = 1024  # For Mistral-7B
MAX_INPUT_LENGTH_LARGE = 2048  # For Llama-2-13B

# PDF formatting
DEFAULT_DPI = 300
DEFAULT_FONTSIZE = 12
MARGIN_POINTS = 72  # 1 inch = 72 points

# Model configurations with version pinning for reproducibility
# Each model includes the HuggingFace model ID and a specific revision (commit hash or tag)
# This ensures consistent results across different runs and environments.
#
# Revisions are based on the latest stable versions as of November 2024:
# - TrOCR: Updated May 2024 - stable OCR model for printed text
# - DiT: Updated Feb 2023 - document layout classification
# - Mistral-7B-v0.2: Updated Jul 2025 - efficient instruction-following model
# - Llama-2-13B: Updated Apr 2024 - larger chat model (gated access required)

MODEL_REVISIONS = {
    "ocr": "c3afae9716f25251a833d4bd3c6a73c61cdb3d63",  # microsoft/trocr-large-printed (May 2024)
    "layout": "4b27643ff7e3e9a0e5e62cea8eb77bb5c9a76f93",  # microsoft/dit-base-finetuned-rvlcdip (Feb 2023)
    "text_small": "b70aa86578567ba3301b21c8a27bea4e8f6d6d61",  # mistralai/Mistral-7B-Instruct-v0.2 (Jul 2025)
    "text_large": "0ba94ac9b9e1d5a0037780667e8b219adde1908c",  # meta-llama/Llama-2-13b-chat-hf (Apr 2024)
}

# Model names (maintains backward compatibility)
# These can be used directly with transformers.pipeline()
# For pinned versions, use get_model_with_revision() helper function
MODELS = {
    "ocr": "microsoft/trocr-large-printed",
    "layout": "microsoft/dit-base-finetuned-rvlcdip",
    "text_small": "mistralai/Mistral-7B-Instruct-v0.2",
    "text_large": "meta-llama/Llama-2-13b-chat-hf",
}


def get_model_with_revision(model_key: str) -> dict:
    """
    Get model configuration with pinned revision for reproducibility.

    Args:
        model_key: Key from MODELS dict ("ocr", "layout", "text_small", "text_large")

    Returns:
        Dictionary with 'model' and 'revision' keys for use with transformers.pipeline()

    Example:
        >>> config = get_model_with_revision("ocr")
        >>> pipeline("image-to-text", **config)
    """
    return {
        "model": MODELS[model_key],
        "revision": MODEL_REVISIONS[model_key],
    }
