"""PDF redaction and anonymization module."""

from pdf_toolkit.redaction.redactor import PDFRedactor
from pdf_toolkit.redaction.patterns import REDACTION_PATTERNS

__all__ = ["PDFRedactor", "REDACTION_PATTERNS"]
