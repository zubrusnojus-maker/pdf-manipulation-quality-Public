"""PDF utility functions for rasterization and optimization."""

import pdf2image
import pikepdf
from tqdm import tqdm

from pdf_toolkit.core.constants import DEFAULT_DPI


def rasterize_pdf(pdf_path, dpi=DEFAULT_DPI):
    """
    Convert PDF pages to images.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rasterization (default: 300)

    Returns:
        List of PIL Image objects
    """
    print(f"Rasterizing PDF at {dpi} DPI...")
    return pdf2image.convert_from_path(pdf_path, dpi=dpi)


def optimize_pdf(input_path, output_path):
    """
    Post-process PDF with pikepdf for optimization.

    Args:
        input_path: Path to input PDF
        output_path: Path to save optimized PDF
    """
    print("Post-processing with pikepdf...")

    source_pdf = pikepdf.Pdf.open(input_path)
    destination_pdf = pikepdf.Pdf.new()

    for page in tqdm(source_pdf.pages, desc="Copying pages"):
        destination_pdf.pages.append(page)

    destination_pdf.save(output_path)
    source_pdf.close()

    print(f"Optimized PDF saved to: {output_path}")
