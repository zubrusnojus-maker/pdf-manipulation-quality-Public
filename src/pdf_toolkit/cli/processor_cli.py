"""CLI for PDF processing with OCR and text manipulation."""

import argparse
from pathlib import Path

from pdf_toolkit.core.processor import PDFProcessor
from pdf_toolkit.utils.pdf_utils import optimize_pdf


def main():
    """Main entry point for PDF processor CLI."""
    parser = argparse.ArgumentParser(description="Process PDFs with OCR and text manipulation")
    parser.add_argument("input", help="Input PDF file path")
    parser.add_argument("-o", "--output", help="Output PDF file path", default=None)
    parser.add_argument(
        "-m",
        "--model-size",
        choices=["small", "large"],
        default="small",
        help="Model size for text processing",
    )
    parser.add_argument(
        "--operation",
        choices=["summarize", "rewrite"],
        default="summarize",
        help="Text manipulation operation",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF rasterization")
    parser.add_argument("--use-layout", action="store_true", help="Enable layout analysis")
    parser.add_argument(
        "--optimize", action="store_true", help="Post-process with pikepdf optimization"
    )

    args = parser.parse_args()

    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"edited_{input_path.name}")

    processor = PDFProcessor(model_size=args.model_size, use_layout=args.use_layout)

    output_path = processor.process_pdf(
        args.input, args.output, operation=args.operation, dpi=args.dpi
    )

    if args.optimize:
        optimized_path = str(Path(output_path).parent / f"optimized_{Path(output_path).name}")
        optimize_pdf(output_path, optimized_path)


if __name__ == "__main__":
    main()
