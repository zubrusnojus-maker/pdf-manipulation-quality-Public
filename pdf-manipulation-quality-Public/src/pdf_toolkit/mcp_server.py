#!/usr/bin/env python3
"""
PDF Toolkit MCP Server

Exposes PDF processing and redaction tools via Model Context Protocol (MCP).
"""

import sys
from pathlib import Path

# FastMCP for creating MCP servers
try:
    from fastmcp import FastMCP
except ImportError:
    print("Installing fastmcp...", file=sys.stderr)
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastmcp"])
    from fastmcp import FastMCP

from pdf_toolkit.core.processor import PDFProcessor
from pdf_toolkit.redaction.redactor import PDFRedactor
from pdf_toolkit.utils.pdf_utils import optimize_pdf, rasterize_pdf

# Create MCP server
mcp = FastMCP("pdf-toolkit")


@mcp.tool()
def redact_pdf(
    input_path: str,
    output_path: str | None = None,
    mapping_output: str | None = None,
    preserve_dates: bool = False,
) -> dict:
    """
    Redact sensitive information from a PDF.

    Detects and replaces PII including:
    - Monetary amounts (e.g., 9,220.51)
    - Account numbers (8+ digits)
    - Dates (DD/MM/YYYY format)
    - UK postcodes (e.g., HA9 6BA)
    - Company names (e.g., ACME LIMITED)
    - Personal names (e.g., Mr John Smith)
    - Street addresses

    Args:
        input_path: Path to input PDF file
        output_path: Path for redacted PDF (default: redacted_<input>.pdf)
        mapping_output: Path to save mappings JSON (optional)
        preserve_dates: If True, don't redact dates

    Returns:
        Dictionary with redaction statistics and mapping info
    """
    input_file = Path(input_path)
    if not input_file.exists():
        return {"error": f"Input file not found: {input_path}"}

    if output_path is None:
        output_path = str(input_file.parent / f"redacted_{input_file.name}")

    try:
        redactor = PDFRedactor()
        stats = redactor.redact_pdf(
            str(input_file),
            output_path,
            preserve_dates=preserve_dates,
        )

        result = {
            "success": True,
            "input": str(input_file),
            "output": output_path,
            "stats": stats,
            "total_redactions": sum(stats.values()) if stats else 0,
        }

        if mapping_output:
            redactor.save_mappings(mapping_output)
            result["mappings_file"] = mapping_output
            result["mappings_count"] = len(redactor.mappings)

        return result
    except Exception as e:
        return {"error": str(e), "input": input_path}


@mcp.tool()
def detect_pii(text: str) -> dict:
    """
    Detect sensitive/PII data in text without modifying it.

    Args:
        text: Text to analyze for sensitive data

    Returns:
        Dictionary with detected items by type
    """
    redactor = PDFRedactor()
    detections = redactor.detect_sensitive_data(text)

    # Group by type
    by_type = {}
    for original, data_type, placeholder in detections:
        if data_type not in by_type:
            by_type[data_type] = []
        by_type[data_type].append(
            {
                "original": original,
                "placeholder": placeholder,
            }
        )

    return {
        "total_detections": len(detections),
        "by_type": by_type,
        "types_found": list(by_type.keys()),
    }


@mcp.tool()
def process_pdf(
    input_path: str,
    output_path: str | None = None,
    operation: str = "summarize",
    model_size: str = "small",
    use_layout: bool = False,
    dpi: int = 300,
) -> dict:
    """
    Process a PDF with OCR and AI text manipulation.

    Uses TrOCR for text extraction and Mistral/Llama for text processing.

    Args:
        input_path: Path to input PDF file
        output_path: Path for processed PDF (default: edited_<input>.pdf)
        operation: "summarize" or "rewrite"
        model_size: "small" (Mistral-7B) or "large" (Llama-2-13B)
        use_layout: Enable layout analysis for better table handling
        dpi: DPI for PDF rasterization (default: 300)

    Returns:
        Dictionary with processing results
    """
    input_file = Path(input_path)
    if not input_file.exists():
        return {"error": f"Input file not found: {input_path}"}

    if output_path is None:
        output_path = str(input_file.parent / f"edited_{input_file.name}")

    try:
        processor = PDFProcessor(model_size=model_size, use_layout=use_layout)
        result_path = processor.process_pdf(
            str(input_file),
            output_path,
            operation=operation,
            dpi=dpi,
        )

        return {
            "success": True,
            "input": str(input_file),
            "output": result_path,
            "operation": operation,
            "model_size": model_size,
            "use_layout": use_layout,
            "dpi": dpi,
        }
    except Exception as e:
        return {"error": str(e), "input": input_path}


@mcp.tool()
def optimize_pdf_file(input_path: str, output_path: str | None = None) -> dict:
    """
    Optimize a PDF file for smaller file size using pikepdf.

    Args:
        input_path: Path to input PDF file
        output_path: Path for optimized PDF (default: optimized_<input>.pdf)

    Returns:
        Dictionary with optimization results
    """
    input_file = Path(input_path)
    if not input_file.exists():
        return {"error": f"Input file not found: {input_path}"}

    if output_path is None:
        output_path = str(input_file.parent / f"optimized_{input_file.name}")

    try:
        import os

        original_size = os.path.getsize(input_file)

        optimize_pdf(str(input_file), output_path)

        new_size = os.path.getsize(output_path)
        reduction = ((original_size - new_size) / original_size) * 100

        return {
            "success": True,
            "input": str(input_file),
            "output": output_path,
            "original_size_bytes": original_size,
            "optimized_size_bytes": new_size,
            "reduction_percent": round(reduction, 2),
        }
    except Exception as e:
        return {"error": str(e), "input": input_path}


@mcp.tool()
def extract_text(input_path: str, dpi: int = 300) -> dict:
    """
    Extract text from a PDF using OCR.

    Args:
        input_path: Path to input PDF file
        dpi: DPI for rasterization (higher = better quality, slower)

    Returns:
        Dictionary with extracted text per page
    """
    input_file = Path(input_path)
    if not input_file.exists():
        return {"error": f"Input file not found: {input_path}"}

    try:
        # Rasterize PDF
        images = rasterize_pdf(str(input_file), dpi=dpi)

        # Extract text
        processor = PDFProcessor()
        processor.load_models()
        texts = processor.extract_text_from_images(images)

        return {
            "success": True,
            "input": str(input_file),
            "page_count": len(texts),
            "pages": [{"page": i + 1, "text": text} for i, text in enumerate(texts)],
        }
    except Exception as e:
        return {"error": str(e), "input": input_path}


@mcp.tool()
def list_pdf_tools() -> dict:
    """
    List all available PDF toolkit tools and their descriptions.

    Returns:
        Dictionary with tool information
    """
    return {
        "tools": [
            {
                "name": "redact_pdf",
                "description": "Redact sensitive information (PII) from PDFs",
                "detects": [
                    "monetary_amounts",
                    "account_numbers",
                    "dates",
                    "postcodes",
                    "company_names",
                    "personal_names",
                    "addresses",
                ],
            },
            {
                "name": "detect_pii",
                "description": "Detect PII in text without modifying it",
            },
            {
                "name": "process_pdf",
                "description": "OCR + AI text manipulation (summarize/rewrite)",
                "models": {"small": "Mistral-7B", "large": "Llama-2-13B"},
            },
            {
                "name": "optimize_pdf_file",
                "description": "Reduce PDF file size with pikepdf",
            },
            {
                "name": "extract_text",
                "description": "Extract text from PDF using TrOCR",
            },
        ],
        "version": "1.0.0",
    }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
