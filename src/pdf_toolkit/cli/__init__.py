"""Command-line interface entry points."""

from pdf_toolkit.cli.processor_cli import main as process_main
from pdf_toolkit.cli.redactor_cli import main as redact_main

__all__ = ["process_main", "redact_main"]
