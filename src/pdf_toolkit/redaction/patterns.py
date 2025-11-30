"""Redaction patterns for sensitive data detection."""

import re

# Default redaction patterns
REDACTION_PATTERNS = {
    "monetary_amount": {
        "pattern": r"\b\d{1,3}(?:,\d{3})*\.\d{2}\b",
        "placeholder_prefix": "Amount",
        "description": "Monetary amounts (e.g., 9,220.51)",
    },
    "account_number": {
        "pattern": r"\b\d{8,}\b",
        "placeholder_prefix": "Account",
        "description": "Account numbers (8+ digits)",
    },
    "date": {
        "pattern": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "placeholder_prefix": "Date",
        "description": "Dates (DD/MM/YYYY, DD-MM-YYYY)",
    },
    "company_name": {
        "pattern": r"\b[A-Z][A-Z\s&]{5,}(?:LTD|LIMITED|INC|CORP|GROUP)\b",
        "placeholder_prefix": "Company",
        "description": "Company names (all caps with suffix)",
    },
    "personal_name": {
        "pattern": r"\b(?:Mr|Mrs|Ms|Dr|Miss)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        "placeholder_prefix": "Person",
        "description": "Personal names with title",
    },
    "postcode": {
        "pattern": r"\b[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}\b",
        "placeholder_prefix": "Postcode",
        "description": "UK postcodes",
    },
    "street_address": {
        "pattern": r"\b\d+\s+[A-Z][A-Za-z\s]+(?:STREET|ROAD|AVENUE|WAY|LANE|DRIVE|CLOSE)\b",
        "placeholder_prefix": "Address",
        "description": "Street addresses",
    },
}


def compile_patterns():
    """Compile all regex patterns for efficient matching."""
    compiled = {}
    for name, config in REDACTION_PATTERNS.items():
        compiled[name] = {
            **config,
            "compiled": re.compile(config["pattern"]),
        }
    return compiled
