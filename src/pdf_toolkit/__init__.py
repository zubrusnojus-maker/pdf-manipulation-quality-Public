"""
PDF Toolkit - Quality-focused PDF processing with OCR and AI text manipulation.

This package provides comprehensive PDF processing capabilities including:
- OCR extraction using TrOCR
- Layout analysis using DiT (Document Image Transformer)
- Text manipulation using Mistral/Llama models
- PDF redaction and anonymization
"""

from pdf_toolkit.core.processor import PDFProcessor
from pdf_toolkit.core.layout import LayoutRegion, LAYOUT_TYPES
from pdf_toolkit.redaction.redactor import PDFRedactor

__version__ = "1.0.0"
__all__ = ["PDFProcessor", "PDFRedactor", "LayoutRegion", "LAYOUT_TYPES"]
