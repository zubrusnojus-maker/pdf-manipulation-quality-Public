"""
Shared constants and configuration for PDF processing toolkit.

This module centralizes all configuration values to avoid magic numbers
and ensure consistency across the codebase.
"""

from pathlib import Path
from typing import Final

# =============================================================================
# File Size and Processing Limits
# =============================================================================
MAX_FILE_SIZE_MB: Final[int] = 100
MAX_FILE_SIZE_BYTES: Final[int] = MAX_FILE_SIZE_MB * 1024 * 1024  # 100MB

MIN_TEXT_LENGTH: Final[int] = 50  # Minimum text length to process
MAX_INPUT_LENGTH_SMALL: Final[int] = 1024  # Token limit for small models
MAX_INPUT_LENGTH_LARGE: Final[int] = 2048  # Token limit for large models

# =============================================================================
# PDF Layout Constants
# =============================================================================
PAGE_MARGIN_PX: Final[int] = 72  # 1 inch margins (72 points per inch)
DPI_DEFAULT: Final[int] = 300  # Default DPI for rasterization
DPI_MIN: Final[int] = 72
DPI_MAX: Final[int] = 600

# =============================================================================
# Model Configuration
# =============================================================================
# OCR Models
OCR_MODEL_TROCR_LARGE: Final[str] = "microsoft/trocr-large-printed"
LAYOUT_MODEL_LAYOUTLMV3: Final[str] = "microsoft/layoutlmv3-base"

# Text Manipulation Models
TEXT_MODEL_MISTRAL: Final[str] = "mistralai/Mistral-7B-Instruct-v0.2"
TEXT_MODEL_LLAMA: Final[str] = "meta-llama/Llama-2-13b-chat-hf"

# Device Configuration
DEVICE_GPU: Final[int] = 0
DEVICE_CPU: Final[int] = -1

# =============================================================================
# Text Generation Parameters
# =============================================================================
MAX_NEW_TOKENS_SMALL: Final[int] = 256
MAX_NEW_TOKENS_LARGE: Final[int] = 512
DEFAULT_TEMPERATURE: Final[float] = 0.7

# =============================================================================
# Colors (RGB tuples, 0-1 range for PyMuPDF)
# =============================================================================
COLOR_BLACK: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0)
COLOR_WHITE: Final[tuple[float, float, float]] = (1.0, 1.0, 1.0)

# =============================================================================
# Redaction Patterns
# =============================================================================
# Sensitive data detection patterns with descriptions
REDACTION_PATTERNS: Final[dict[str, dict[str, str]]] = {
    # Financial patterns
    "monetary_amount": {
        "pattern": r"\b\d{1,3}(?:,\d{3})*\.\d{2}\b",
        "description": "Monetary amounts (e.g., 9,220.51)",
        "prefix": "Amount_",
    },
    "monetary_no_comma": {
        "pattern": r"\b\d{4,}\.\d{2}\b",
        "description": "Monetary amounts without commas (e.g., 1000.50)",
        "prefix": "Amount_",
    },
    "account_number": {
        "pattern": r"\b\d{8,}\b",
        "description": "Account numbers (8+ digits)",
        "prefix": "Account_",
    },
    "credit_card": {
        "pattern": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "description": "Credit card numbers",
        "prefix": "Card_",
    },
    # Personal identifiers
    "email": {
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "description": "Email addresses",
        "prefix": "Email_",
    },
    "phone_uk": {
        "pattern": r"\b(?:\+44|0)\s?(?:\d{4}\s?\d{6}|\d{3}\s?\d{3}\s?\d{4})\b",
        "description": "UK phone numbers",
        "prefix": "Phone_",
    },
    "phone_us": {
        "pattern": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "description": "US phone numbers",
        "prefix": "Phone_",
    },
    "phone_intl": {
        "pattern": r"\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b",
        "description": "International phone numbers",
        "prefix": "Phone_",
    },
    "ssn": {
        "pattern": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "description": "US Social Security Numbers",
        "prefix": "SSN_",
    },
    "nino": {
        "pattern": r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b",
        "description": "UK National Insurance Numbers",
        "prefix": "NINO_",
    },
    # Dates
    "date_slash": {
        "pattern": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        "description": "Dates with slashes (DD/MM/YYYY)",
        "prefix": "Date_",
    },
    "date_dash": {
        "pattern": r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
        "description": "Dates with dashes",
        "prefix": "Date_",
    },
    "date_iso": {
        "pattern": r"\b\d{4}-\d{2}-\d{2}\b",
        "description": "ISO format dates (YYYY-MM-DD)",
        "prefix": "Date_",
    },
    # Names and entities
    "personal_name": {
        "pattern": r"\b(?:Mr|Mrs|Ms|Dr|Miss|Prof|Sir|Madam|Hon)\.?\s+[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)*\b",
        "description": "Personal names with titles",
        "prefix": "Person_",
    },
    "company_name": {
        "pattern": r"\b[A-Z][A-Z\s&]{3,}(?:LTD|LIMITED|INC|CORP|GROUP|PLC|LLC|AG|GMBH)\b",
        "description": "Company names",
        "prefix": "Company_",
    },
    # Addresses
    "postcode_uk": {
        "pattern": r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b",
        "description": "UK postcodes",
        "prefix": "Postcode_",
    },
    "zipcode_us": {
        "pattern": r"\b\d{5}(?:-\d{4})?\b",
        "description": "US ZIP codes",
        "prefix": "Zip_",
    },
    "street_address": {
        "pattern": r"\b\d+\s+[A-Z][A-Za-z\s]+(?:STREET|ROAD|AVENUE|AVE|WAY|LANE|DRIVE|DR|CLOSE|COURT|CT|PLACE|PL|BOULEVARD|BLVD|TERRACE|CIRCLE|CIR)\b",
        "description": "Street addresses",
        "prefix": "Address_",
    },
    # IDs and references
    "passport": {
        "pattern": r"\b[A-Z]{1,2}\d{6,9}\b",
        "description": "Passport numbers",
        "prefix": "Passport_",
    },
    "iban": {
        "pattern": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b",
        "description": "IBAN numbers",
        "prefix": "IBAN_",
    },
    "ip_address": {
        "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "description": "IP addresses",
        "prefix": "IP_",
    },
}

# Patterns that should be excluded from certain matches (to avoid false positives)
PATTERN_EXCLUSIONS: Final[dict[str, list[str]]] = {
    "account_number": ["monetary_amount", "date_slash", "date_dash", "phone_uk", "phone_us"],
    "zipcode_us": ["monetary_amount"],
}

# =============================================================================
# Mapping File Security
# =============================================================================
MAPPING_FILE_PERMISSIONS: Final[int] = 0o600  # Owner read/write only
MAPPING_FILE_WARNING: Final[str] = """
WARNING: This mapping file contains the original sensitive data that was redacted.
It should be treated with the same security as the original document.
Store securely and delete when no longer needed.
"""

# =============================================================================
# Supported File Types
# =============================================================================
SUPPORTED_PDF_EXTENSIONS: Final[set[str]] = {".pdf"}
VALID_OPERATIONS: Final[set[str]] = {"summarize", "rewrite"}
VALID_MODEL_SIZES: Final[set[str]] = {"small", "large"}

# =============================================================================
# Exit Codes
# =============================================================================
EXIT_SUCCESS: Final[int] = 0
EXIT_FILE_NOT_FOUND: Final[int] = 1
EXIT_INVALID_PDF: Final[int] = 2
EXIT_PROCESSING_ERROR: Final[int] = 3
EXIT_INVALID_ARGUMENTS: Final[int] = 4
