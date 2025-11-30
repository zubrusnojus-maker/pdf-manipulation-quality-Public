#!/usr/bin/env python3
"""
PDF Data Anonymization/Redaction Tool

This is a backwards-compatible wrapper. The implementation has moved to:
    src/pdf_toolkit/redaction/redactor.py

Usage:
    python pdf_redactor.py input.pdf -o output.pdf
    python pdf_redactor.py input.pdf --mapping-output mappings.json
"""

import sys
import warnings
from pathlib import Path

# Deprecation warning
warnings.warn(
    "pdf_redactor.py is deprecated and will be removed in v2.0.0. "
    "Please use 'from pdf_toolkit.redaction.redactor import PDFRedactor' or "
    "the 'pdf-redact' CLI command instead. "
    "See MIGRATION_GUIDE.md for details.",
    DeprecationWarning,
    stacklevel=2,
)

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Re-export from new package location for backwards compatibility
from pdf_toolkit.cli.redactor_cli import main
from pdf_toolkit.redaction.patterns import REDACTION_PATTERNS
from pdf_toolkit.redaction.redactor import PDFRedactor

__all__ = ["PDFRedactor", "REDACTION_PATTERNS", "main"]

if __name__ == "__main__":
    main()
