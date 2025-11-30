"""Constants used throughout the PDF processing pipeline."""

# Text processing
MIN_TEXT_LENGTH = 50
MAX_INPUT_LENGTH_SMALL = 1024  # For Mistral-7B
MAX_INPUT_LENGTH_LARGE = 2048  # For Llama-2-13B

# PDF formatting
DEFAULT_DPI = 300
DEFAULT_FONTSIZE = 12
MARGIN_POINTS = 72  # 1 inch = 72 points

# Model configurations
MODELS = {
    "ocr": "microsoft/trocr-large-printed",
    "layout": "microsoft/dit-base-finetuned-rvlcdip",
    "text_small": "mistralai/Mistral-7B-Instruct-v0.2",
    "text_large": "meta-llama/Llama-2-13b-chat-hf",
}
