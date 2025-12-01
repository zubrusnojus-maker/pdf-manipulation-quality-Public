"""CLI for PDF redaction and anonymization."""

import argparse
from pathlib import Path

from pdf_toolkit.redaction.redactor import PDFRedactor


def main():
    """Main entry point for PDF redactor CLI."""
    parser = argparse.ArgumentParser(description="Anonymize PDF data while preserving formatting")
    parser.add_argument("input", help="Input PDF file path")
    parser.add_argument("-o", "--output", help="Output PDF file path", default=None)
    parser.add_argument(
        "--mapping-output", help="Path to save mapping JSON (original → placeholder)", default=None
    )
    parser.add_argument("--preserve-dates", action="store_true", help="Do not redact dates")

    args = parser.parse_args()

    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"redacted_{input_path.name}")

    if args.mapping_output is None:
        output_path = Path(args.output)
        args.mapping_output = str(output_path.parent / f"{output_path.stem}_mappings.json")

    redactor = PDFRedactor()
    redactor.redact_pdf(args.input, args.output)
    redactor.save_mappings(args.mapping_output)

    print("\nComplete! Files created:")
    print(f"  - Redacted PDF: {args.output}")
    print(f"  - Mappings: {args.mapping_output}")


if __name__ == "__main__":
    main()
