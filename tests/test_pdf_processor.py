"""
Tests for pdf_processor.py module.

These tests verify the PDF processing pipeline including initialization,
model loading, OCR, text manipulation, and output generation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf_processor import PDFProcessor, main, post_process_with_pikepdf
from pdf_utils import PDFProcessingError, PDFValidationError


class TestPDFProcessorInit:
    """Tests for PDFProcessor initialization."""

    def test_default_initialization(self) -> None:
        """Should initialize with default settings."""
        processor = PDFProcessor()
        assert processor.model_size == "small"
        assert processor.use_layout is False
        assert processor._models_loaded is False

    def test_large_model_size(self) -> None:
        """Should accept large model size."""
        processor = PDFProcessor(model_size="large")
        assert processor.model_size == "large"

    def test_with_layout(self) -> None:
        """Should accept use_layout option."""
        processor = PDFProcessor(use_layout=True)
        assert processor.use_layout is True

    def test_invalid_model_size(self) -> None:
        """Should reject invalid model size."""
        with pytest.raises(ValueError, match="Invalid model_size"):
            PDFProcessor(model_size="invalid")


class TestLoadPDF:
    """Tests for PDF loading."""

    def test_load_valid_pdf(self, sample_pdf: Path) -> None:
        """Should load valid PDF."""
        processor = PDFProcessor()
        doc = processor.load_pdf(sample_pdf)
        assert len(doc) > 0
        doc.close()

    def test_load_missing_pdf(self, temp_dir: Path) -> None:
        """Should raise error for missing file."""
        processor = PDFProcessor()
        with pytest.raises(FileNotFoundError):
            processor.load_pdf(temp_dir / "missing.pdf")

    def test_load_invalid_pdf(self, invalid_pdf: Path) -> None:
        """Should raise error for invalid PDF."""
        processor = PDFProcessor()
        with pytest.raises(PDFValidationError):
            processor.load_pdf(invalid_pdf)


class TestRasterizePDF:
    """Tests for PDF rasterization."""

    def test_rasterize_with_default_dpi(
        self, sample_pdf: Path, mock_pdf2image: MagicMock
    ) -> None:
        """Should rasterize with default DPI."""
        processor = PDFProcessor()
        images = processor.rasterize_pdf(sample_pdf)
        assert len(images) > 0

    def test_rasterize_with_custom_dpi(
        self, sample_pdf: Path, mock_pdf2image: MagicMock
    ) -> None:
        """Should rasterize with custom DPI."""
        processor = PDFProcessor()
        images = processor.rasterize_pdf(sample_pdf, dpi=150)
        mock_pdf2image.assert_called_with(str(sample_pdf), dpi=150)

    def test_rasterize_invalid_dpi(self, sample_pdf: Path) -> None:
        """Should reject invalid DPI."""
        processor = PDFProcessor()
        with pytest.raises(ValueError):
            processor.rasterize_pdf(sample_pdf, dpi=50)


class TestExtractText:
    """Tests for text extraction."""

    def test_extract_without_models(self) -> None:
        """Should raise error if models not loaded."""
        processor = PDFProcessor()
        with pytest.raises(RuntimeError, match="not loaded"):
            processor.extract_text_from_images([MagicMock()])

    def test_extract_with_mock_pipeline(self, mock_ocr_pipeline: MagicMock) -> None:
        """Should extract text with mocked pipeline."""
        processor = PDFProcessor()
        processor.ocr_pipe = mock_ocr_pipeline.return_value

        mock_image = MagicMock()
        texts = processor.extract_text_from_images([mock_image])

        assert len(texts) == 1


class TestManipulateText:
    """Tests for text manipulation."""

    def test_manipulate_without_models(self) -> None:
        """Should raise error if models not loaded."""
        processor = PDFProcessor()
        with pytest.raises(RuntimeError, match="not loaded"):
            processor.manipulate_text(["test"], operation="summarize")

    def test_invalid_operation(self) -> None:
        """Should reject invalid operation."""
        processor = PDFProcessor()
        processor.summariser = MagicMock()
        with pytest.raises(ValueError, match="Invalid operation"):
            processor.manipulate_text(["test"], operation="invalid")

    def test_skip_short_text(self, mock_pipeline: MagicMock) -> None:
        """Should skip text below minimum length."""
        processor = PDFProcessor()
        processor.summariser = mock_pipeline.return_value

        short_text = "Hi"
        result = processor.manipulate_text([short_text])

        assert result[0] == short_text

    def test_summarize_operation(self, mock_pipeline: MagicMock) -> None:
        """Should format prompt for summarize operation."""
        processor = PDFProcessor()
        processor.summariser = mock_pipeline.return_value

        long_text = "This is a sufficiently long text for testing. " * 5
        result = processor.manipulate_text([long_text], operation="summarize")

        # Verify pipeline was called
        assert len(result) == 1

    def test_rewrite_operation(self, mock_pipeline: MagicMock) -> None:
        """Should format prompt for rewrite operation."""
        processor = PDFProcessor()
        processor.summariser = mock_pipeline.return_value

        long_text = "This is a sufficiently long text for testing. " * 5
        result = processor.manipulate_text([long_text], operation="rewrite")

        assert len(result) == 1


class TestFormatPrompt:
    """Tests for prompt formatting."""

    def test_summarize_prompt(self) -> None:
        """Should format summarize prompt correctly."""
        processor = PDFProcessor()
        prompt = processor._format_prompt("test text", "summarize")

        assert "[INST]" in prompt
        assert "Summarize" in prompt
        assert "test text" in prompt

    def test_rewrite_prompt(self) -> None:
        """Should format rewrite prompt correctly."""
        processor = PDFProcessor()
        prompt = processor._format_prompt("test text", "rewrite")

        assert "[INST]" in prompt
        assert "Rewrite" in prompt
        assert "test text" in prompt


class TestReinsertText:
    """Tests for text reinsertion."""

    def test_reinsert_text(self, sample_pdf: Path) -> None:
        """Should insert text into PDF pages."""
        import fitz

        processor = PDFProcessor()
        doc = fitz.open(str(sample_pdf))

        texts = ["Inserted text"]
        processor.reinsert_text(doc, texts)

        # Verify text was inserted
        text = doc[0].get_text()
        assert "Inserted" in text
        doc.close()

    def test_reinsert_more_texts_than_pages(self, sample_pdf: Path) -> None:
        """Should handle more texts than pages."""
        import fitz

        processor = PDFProcessor()
        doc = fitz.open(str(sample_pdf))

        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        processor.reinsert_text(doc, texts)  # Should not crash
        doc.close()


class TestProcessPDF:
    """Tests for full processing pipeline."""

    def test_process_creates_output(
        self,
        sample_pdf: Path,
        temp_dir: Path,
        mock_pipeline: MagicMock,
        mock_pdf2image: MagicMock,
    ) -> None:
        """Should create output file."""
        output_path = temp_dir / "output.pdf"

        processor = PDFProcessor()
        processor.ocr_pipe = mock_pipeline.return_value
        processor.summariser = mock_pipeline.return_value
        processor._models_loaded = True

        result = processor.process_pdf(sample_pdf, output_path)

        assert output_path.exists()
        assert result == output_path

    def test_process_invalid_input(self, temp_dir: Path) -> None:
        """Should raise error for invalid input."""
        processor = PDFProcessor()
        output_path = temp_dir / "output.pdf"

        with pytest.raises(FileNotFoundError):
            processor.process_pdf(temp_dir / "missing.pdf", output_path)


class TestCleanup:
    """Tests for resource cleanup."""

    def test_cleanup_releases_resources(self) -> None:
        """Should release model resources."""
        processor = PDFProcessor()
        processor.ocr_pipe = MagicMock()
        processor.summariser = MagicMock()
        processor._models_loaded = True

        with patch("torch.cuda.is_available", return_value=False):
            processor.cleanup()

        assert processor.ocr_pipe is None
        assert processor.summariser is None
        assert processor._models_loaded is False


class TestPostProcessWithPikepdf:
    """Tests for pikepdf post-processing."""

    def test_post_process_creates_output(
        self, sample_pdf: Path, temp_dir: Path
    ) -> None:
        """Should create optimized output."""
        output_path = temp_dir / "optimized.pdf"

        result = post_process_with_pikepdf(sample_pdf, output_path)

        assert output_path.exists()
        assert result == output_path

    def test_post_process_invalid_input(self, temp_dir: Path) -> None:
        """Should raise error for invalid input."""
        output_path = temp_dir / "optimized.pdf"

        with pytest.raises(PDFProcessingError):
            post_process_with_pikepdf(temp_dir / "missing.pdf", output_path)


class TestMain:
    """Tests for CLI main function."""

    def test_main_help(self) -> None:
        """Should show help and exit."""
        with patch("sys.argv", ["pdf_processor.py", "-h"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_file_not_found(self, temp_dir: Path) -> None:
        """Should return error for missing file."""
        with patch("sys.argv", ["pdf_processor.py", str(temp_dir / "missing.pdf")]):
            result = main()
            assert result != 0

    def test_main_invalid_dpi(self, sample_pdf: Path) -> None:
        """Should return error for invalid DPI."""
        with patch("sys.argv", ["pdf_processor.py", str(sample_pdf), "--dpi", "50"]):
            result = main()
            assert result != 0


class TestModelLoading:
    """Tests for model loading behavior."""

    def test_models_not_loaded_initially(self) -> None:
        """Models should not be loaded on init."""
        processor = PDFProcessor()
        assert processor._models_loaded is False
        assert processor.ocr_pipe is None
        assert processor.summariser is None

    def test_double_load_skipped(self, mock_pipeline: MagicMock) -> None:
        """Should skip loading if already loaded."""
        processor = PDFProcessor()
        processor._models_loaded = True
        processor.ocr_pipe = MagicMock()
        processor.summariser = MagicMock()

        # Should not call pipeline again
        processor.load_models()
        mock_pipeline.assert_not_called()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_multi_page_processing(
        self,
        multi_page_pdf: Path,
        temp_dir: Path,
        mock_pipeline: MagicMock,
        mock_pdf2image: MagicMock,
    ) -> None:
        """Should handle multi-page PDFs."""
        # Mock multiple pages
        mock_images = [MagicMock() for _ in range(5)]
        mock_pdf2image.return_value = mock_images

        output_path = temp_dir / "output.pdf"
        processor = PDFProcessor()
        processor.ocr_pipe = mock_pipeline.return_value
        processor.summariser = mock_pipeline.return_value
        processor._models_loaded = True

        result = processor.process_pdf(multi_page_pdf, output_path)
        assert result == output_path

    def test_empty_pdf_handling(
        self,
        empty_pdf: Path,
        temp_dir: Path,
        mock_pipeline: MagicMock,
        mock_pdf2image: MagicMock,
    ) -> None:
        """Should handle empty PDFs."""
        output_path = temp_dir / "output.pdf"
        processor = PDFProcessor()
        processor.ocr_pipe = mock_pipeline.return_value
        processor.summariser = mock_pipeline.return_value
        processor._models_loaded = True

        # Should not crash
        result = processor.process_pdf(empty_pdf, output_path)
        assert result == output_path
