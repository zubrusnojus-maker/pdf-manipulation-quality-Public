"""
PDF Processing Script with OCR and Text Manipulation

This is a backwards-compatible wrapper. The implementation has moved to:
    src/pdf_toolkit/core/processor.py

Usage:
    python pdf_processor.py input.pdf -o output.pdf --use-layout
"""

import sys
import warnings
from pathlib import Path

# Deprecation warning
warnings.warn(
    "pdf_processor.py is deprecated and will be removed in v2.0.0. "
    "Please use 'from pdf_toolkit.core.processor import PDFProcessor' or "
    "the 'pdf-process' CLI command instead. "
    "See MIGRATION_GUIDE.md for details.",
    DeprecationWarning,
    stacklevel=2,
)

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Re-export from new package location for backwards compatibility
from pdf_toolkit.cli.processor_cli import main
from pdf_toolkit.core.constants import MIN_TEXT_LENGTH
from pdf_toolkit.core.layout import LAYOUT_TYPES, LayoutRegion
from pdf_toolkit.core.processor import PDFProcessor
from pdf_toolkit.utils.pdf_utils import optimize_pdf as post_process_with_pikepdf

__all__ = [
    "PDFProcessor",
    "LayoutRegion",
    "LAYOUT_TYPES",
    "MIN_TEXT_LENGTH",
    "post_process_with_pikepdf",
    "main",
]

if __name__ == "__main__":
    main()
