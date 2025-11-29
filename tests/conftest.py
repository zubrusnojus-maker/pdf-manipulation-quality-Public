"""
Pytest fixtures and configuration for PDF Processing Toolkit tests.

This module provides shared fixtures for creating test PDFs, mocking
external dependencies (ML models, GPU), and managing test resources.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

if TYPE_CHECKING:
    import fitz


# =============================================================================
# Test Data Constants
# =============================================================================

SAMPLE_TEXT = "This is a sample text for testing PDF processing functionality."
SAMPLE_TEXT_LONG = SAMPLE_TEXT * 20  # ~1000 characters

SENSITIVE_DATA_SAMPLES = {
    "email": "john.doe@example.com",
    "phone_uk": "+44 1234 567890",
    "phone_us": "(555) 123-4567",
    "ssn": "123-45-6789",
    "credit_card": "4111-1111-1111-1111",
    "monetary": "1,234.56",
    "date": "25/12/2024",
    "postcode_uk": "SW1A 1AA",
    "name": "Mr John Smith",
    "company": "ACME CORPORATION LTD",
}


# =============================================================================
# PDF Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="pdf_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a minimal valid PDF file for testing."""
    import fitz

    pdf_path = temp_dir / "sample.pdf"
    doc = fitz.open()

    # Create a page with sample text
    page = doc.new_page()
    page.insert_text((72, 72), SAMPLE_TEXT, fontsize=12)

    doc.save(str(pdf_path))
    doc.close()

    yield pdf_path


@pytest.fixture
def multi_page_pdf(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a multi-page PDF for testing."""
    import fitz

    pdf_path = temp_dir / "multi_page.pdf"
    doc = fitz.open()

    for i in range(5):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}: {SAMPLE_TEXT}", fontsize=12)

    doc.save(str(pdf_path))
    doc.close()

    yield pdf_path


@pytest.fixture
def sensitive_data_pdf(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a PDF containing various sensitive data patterns."""
    import fitz

    pdf_path = temp_dir / "sensitive.pdf"
    doc = fitz.open()
    page = doc.new_page()

    y_position = 72
    for data_type, value in SENSITIVE_DATA_SAMPLES.items():
        page.insert_text((72, y_position), f"{data_type}: {value}", fontsize=12)
        y_position += 20

    doc.save(str(pdf_path))
    doc.close()

    yield pdf_path


@pytest.fixture
def empty_pdf(temp_dir: Path) -> Generator[Path, None, None]:
    """Create an empty PDF (no text content)."""
    import fitz

    pdf_path = temp_dir / "empty.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    yield pdf_path


@pytest.fixture
def invalid_pdf(temp_dir: Path) -> Generator[Path, None, None]:
    """Create an invalid PDF file (not a real PDF)."""
    pdf_path = temp_dir / "invalid.pdf"
    pdf_path.write_text("This is not a valid PDF file")
    yield pdf_path


@pytest.fixture
def large_pdf(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a PDF with long text content."""
    import fitz

    pdf_path = temp_dir / "large.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), SAMPLE_TEXT_LONG, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    yield pdf_path


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_torch_cuda() -> Generator[MagicMock, None, None]:
    """Mock torch.cuda.is_available to return False (CPU mode)."""
    with patch("torch.cuda.is_available", return_value=False) as mock:
        yield mock


@pytest.fixture
def mock_torch_cuda_available() -> Generator[MagicMock, None, None]:
    """Mock torch.cuda.is_available to return True (GPU mode)."""
    with patch("torch.cuda.is_available", return_value=True) as mock:
        yield mock


@pytest.fixture
def mock_pipeline() -> Generator[MagicMock, None, None]:
    """Mock transformers.pipeline to avoid model downloads."""
    mock_result = [{"generated_text": "Mocked summarization output"}]

    with patch("transformers.pipeline") as mock:
        mock_instance = MagicMock()
        mock_instance.return_value = mock_result
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_ocr_pipeline() -> Generator[MagicMock, None, None]:
    """Mock OCR pipeline with realistic output."""
    mock_result = [{"generated_text": SAMPLE_TEXT}]

    with patch("transformers.pipeline") as mock:
        mock_instance = MagicMock()
        mock_instance.return_value = mock_result
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_pdf2image() -> Generator[MagicMock, None, None]:
    """Mock pdf2image.convert_from_path."""
    mock_image = MagicMock()
    mock_image.size = (612, 792)  # Letter size in pixels at 72 DPI

    with patch("pdf2image.convert_from_path") as mock:
        mock.return_value = [mock_image]
        yield mock


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def clean_environment() -> Generator[None, None, None]:
    """Ensure a clean environment for testing."""
    # Store original environment
    original_env = os.environ.copy()

    yield

    # Restore environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def output_dir(temp_dir: Path) -> Path:
    """Create an output directory for test results."""
    output = temp_dir / "output"
    output.mkdir()
    return output


# =============================================================================
# Utility Functions
# =============================================================================


def create_test_pdf(
    path: Path,
    text: str = SAMPLE_TEXT,
    num_pages: int = 1,
) -> Path:
    """
    Create a test PDF with specified content.

    Args:
        path: Path to create the PDF
        text: Text content for each page
        num_pages: Number of pages to create

    Returns:
        Path to the created PDF
    """
    import fitz

    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}: {text}", fontsize=12)

    doc.save(str(path))
    doc.close()
    return path


def get_pdf_text(path: Path) -> str:
    """
    Extract all text from a PDF.

    Args:
        path: Path to the PDF

    Returns:
        Concatenated text from all pages
    """
    import fitz

    doc = fitz.open(str(path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def get_pdf_page_count(path: Path) -> int:
    """
    Get the number of pages in a PDF.

    Args:
        path: Path to the PDF

    Returns:
        Number of pages
    """
    import fitz

    doc = fitz.open(str(path))
    count = len(doc)
    doc.close()
    return count
