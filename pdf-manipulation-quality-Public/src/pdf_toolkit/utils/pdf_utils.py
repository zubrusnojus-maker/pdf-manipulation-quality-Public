"""PDF utility functions for rasterization and optimization."""

import pdf2image
import pikepdf
from tqdm import tqdm

from pdf_toolkit.core.constants import DEFAULT_DPI

try:
    import fitz
    from PIL import Image
except Exception:
    fitz = None
    Image = None


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
    try:
        return pdf2image.convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"pdf2image failed ({e}), falling back to PyMuPDF rendering...")
        if fitz is None or Image is None:
            raise

        images = []
        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
        doc.close()
        return images


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
