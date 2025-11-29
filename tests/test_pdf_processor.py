"""Tests for PDF processing functionality."""

import pytest
from pdf_processor import MIN_TEXT_LENGTH, PDFProcessor


class TestPDFProcessor:
    """Test suite for PDFProcessor class."""

    def test_init_small_model(self):
        """Test PDFProcessor initialization with small model."""
        processor = PDFProcessor(model_size="small", use_layout=False)
        assert processor.model_size == "small"
        assert processor.use_layout is False
        assert processor.ocr_pipe is None
        assert processor.summariser is None

    def test_init_large_model(self):
        """Test PDFProcessor initialization with large model."""
        processor = PDFProcessor(model_size="large", use_layout=True)
        assert processor.model_size == "large"
        assert processor.use_layout is True

    def test_constants(self):
        """Test module constants."""
        assert MIN_TEXT_LENGTH == 50
        assert isinstance(MIN_TEXT_LENGTH, int)
        assert MIN_TEXT_LENGTH > 0


@pytest.mark.parametrize("model_size", ["small", "large"])
def test_model_size_parameter(model_size):
    """Test different model sizes can be initialized."""
    processor = PDFProcessor(model_size=model_size)
    assert processor.model_size == model_size


@pytest.mark.parametrize("use_layout", [True, False])
def test_layout_parameter(use_layout):
    """Test layout analysis parameter."""
    processor = PDFProcessor(use_layout=use_layout)
    assert processor.use_layout == use_layout
