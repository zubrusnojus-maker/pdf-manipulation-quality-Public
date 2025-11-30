"""
PDF Processing Script with OCR and Text Manipulation

This is a backwards-compatible wrapper. The implementation has moved to:
    src/pdf_toolkit/core/processor.py

Usage:
    python pdf_processor.py input.pdf -o output.pdf --use-layout
"""

import sys
from pathlib import Path

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Re-export from new package location for backwards compatibility
from pdf_toolkit.core.processor import PDFProcessor
from pdf_toolkit.core.layout import LayoutRegion, LAYOUT_TYPES
from pdf_toolkit.core.constants import MIN_TEXT_LENGTH
from pdf_toolkit.utils.pdf_utils import optimize_pdf as post_process_with_pikepdf
from pdf_toolkit.cli.processor_cli import main

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
