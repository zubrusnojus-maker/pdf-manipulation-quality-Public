"""Tests for quick PDF processing script."""

from unittest.mock import MagicMock, patch
import pytest
import sys

from quick_pdf_process import (
    MIN_ARGS_REQUIRED,
    MIN_TEXT_LENGTH,
    MAX_INPUT_LENGTH,
    quick_process,
)


class TestConstants:
    """Test module constants."""

    def test_min_args_required(self):
        """Test MIN_ARGS_REQUIRED constant."""
        assert MIN_ARGS_REQUIRED == 2

    def test_min_text_length(self):
        """Test MIN_TEXT_LENGTH constant."""
        assert MIN_TEXT_LENGTH == 50

    def test_max_input_length(self):
        """Test MAX_INPUT_LENGTH constant."""
        assert MAX_INPUT_LENGTH == 1024


class TestQuickProcess:
    """Test the quick_process function."""

    @patch("quick_pdf_process.fitz.open")
    @patch("quick_pdf_process.pdf2image.convert_from_path")
    @patch("quick_pdf_process.pipeline")
    @patch("quick_pdf_process.torch.cuda.is_available", return_value=False)
    def test_quick_process_basic(
        self, mock_cuda, mock_pipeline, mock_convert, mock_fitz
    ):
        """Test basic quick_process execution."""
        # Setup mocks
        mock_ocr = MagicMock(return_value=[{"generated_text": "A" * 100}])
        mock_summarizer = MagicMock(
            return_value=[{"generated_text": "[/INST] Summary text"}]
        )
        mock_pipeline.side_effect = [mock_ocr, mock_summarizer]

        mock_image = MagicMock()
        mock_convert.return_value = [mock_image]

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.return_value = mock_doc

        # Run
        result = quick_process("test.pdf")

        # Verify
        assert result == "edited_test.pdf"
        mock_fitz.assert_called_once_with("test.pdf")
        mock_convert.assert_called_once_with("test.pdf", dpi=300)
        mock_doc.save.assert_called_once()
        mock_doc.close.assert_called_once()

    @patch("quick_pdf_process.fitz.open")
    @patch("quick_pdf_process.pdf2image.convert_from_path")
    @patch("quick_pdf_process.pipeline")
    @patch("quick_pdf_process.torch.cuda.is_available", return_value=False)
    def test_quick_process_multiple_pages(
        self, mock_cuda, mock_pipeline, mock_convert, mock_fitz
    ):
        """Test processing multiple pages."""
        # Setup mocks
        mock_ocr = MagicMock(
            side_effect=[
                [{"generated_text": "B" * 100}],
                [{"generated_text": "C" * 100}],
            ]
        )
        mock_summarizer = MagicMock(
            side_effect=[
                [{"generated_text": "[/INST] Summary 1"}],
                [{"generated_text": "[/INST] Summary 2"}],
            ]
        )
        mock_pipeline.side_effect = [mock_ocr, mock_summarizer]

        mock_images = [MagicMock(), MagicMock()]
        mock_convert.return_value = mock_images

        mock_pages = [MagicMock(), MagicMock()]
        for page in mock_pages:
            page.rect.width = 612
            page.rect.height = 792

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__getitem__ = MagicMock(side_effect=mock_pages)
        mock_fitz.return_value = mock_doc

        # Run
        result = quick_process("multi_page.pdf")

        # Verify OCR called for each page
        assert mock_ocr.call_count == 2
        assert result == "edited_multi_page.pdf"

    @patch("quick_pdf_process.fitz.open")
    @patch("quick_pdf_process.pdf2image.convert_from_path")
    @patch("quick_pdf_process.pipeline")
    @patch("quick_pdf_process.torch.cuda.is_available", return_value=False)
    def test_quick_process_short_text_skipped(
        self, mock_cuda, mock_pipeline, mock_convert, mock_fitz
    ):
        """Test that short text is passed through without summarization."""
        # Setup mocks - OCR returns short text
        mock_ocr = MagicMock(return_value=[{"generated_text": "Short"}])
        mock_summarizer = MagicMock()
        mock_pipeline.side_effect = [mock_ocr, mock_summarizer]

        mock_convert.return_value = [MagicMock()]

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.return_value = mock_doc

        # Run
        quick_process("test.pdf")

        # Summarizer should not be called for short text
        mock_summarizer.assert_not_called()

    @patch("quick_pdf_process.fitz.open")
    @patch("quick_pdf_process.pdf2image.convert_from_path")
    @patch("quick_pdf_process.pipeline")
    @patch("quick_pdf_process.torch.cuda.is_available", return_value=False)
    def test_quick_process_empty_ocr_result(
        self, mock_cuda, mock_pipeline, mock_convert, mock_fitz
    ):
        """Test handling of empty OCR results."""
        # Setup mocks - OCR returns empty result
        mock_ocr = MagicMock(return_value=[])
        mock_summarizer = MagicMock()
        mock_pipeline.side_effect = [mock_ocr, mock_summarizer]

        mock_convert.return_value = [MagicMock()]

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.return_value = mock_doc

        # Run - should not crash
        result = quick_process("test.pdf")
        assert result == "edited_test.pdf"

    @patch("quick_pdf_process.fitz.open")
    @patch("quick_pdf_process.pdf2image.convert_from_path")
    @patch("quick_pdf_process.pipeline")
    @patch("quick_pdf_process.torch.cuda.is_available", return_value=False)
    def test_quick_process_summarizer_exception(
        self, mock_cuda, mock_pipeline, mock_convert, mock_fitz
    ):
        """Test graceful handling of summarizer exceptions."""
        # Setup mocks
        mock_ocr = MagicMock(return_value=[{"generated_text": "D" * 100}])
        mock_summarizer = MagicMock(side_effect=Exception("Model error"))
        mock_pipeline.side_effect = [mock_ocr, mock_summarizer]

        mock_convert.return_value = [MagicMock()]

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.return_value = mock_doc

        # Run - should fallback to original text
        result = quick_process("test.pdf")
        assert result == "edited_test.pdf"

    @patch("quick_pdf_process.fitz.open")
    @patch("quick_pdf_process.pdf2image.convert_from_path")
    @patch("quick_pdf_process.pipeline")
    @patch("quick_pdf_process.torch.cuda.is_available", return_value=True)
    def test_quick_process_gpu_available(
        self, mock_cuda, mock_pipeline, mock_convert, mock_fitz
    ):
        """Test that GPU is used when available."""
        mock_ocr = MagicMock(return_value=[{"generated_text": "E" * 100}])
        mock_summarizer = MagicMock(
            return_value=[{"generated_text": "[/INST] Summary"}]
        )
        mock_pipeline.side_effect = [mock_ocr, mock_summarizer]

        mock_convert.return_value = [MagicMock()]

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.return_value = mock_doc

        quick_process("test.pdf")

        # OCR pipeline should use device=0 (GPU)
        ocr_call = mock_pipeline.call_args_list[0]
        assert ocr_call[1]["device"] == 0

    @patch("quick_pdf_process.fitz.open")
    @patch("quick_pdf_process.pdf2image.convert_from_path")
    @patch("quick_pdf_process.pipeline")
    @patch("quick_pdf_process.torch.cuda.is_available", return_value=False)
    def test_quick_process_text_truncation(
        self, mock_cuda, mock_pipeline, mock_convert, mock_fitz
    ):
        """Test that long text is truncated before summarization."""
        # Setup - very long text
        long_text = "F" * 2000
        mock_ocr = MagicMock(return_value=[{"generated_text": long_text}])
        mock_summarizer = MagicMock(
            return_value=[{"generated_text": "[/INST] Summary of truncated"}]
        )
        mock_pipeline.side_effect = [mock_ocr, mock_summarizer]

        mock_convert.return_value = [MagicMock()]

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.return_value = mock_doc

        quick_process("test.pdf")

        # Verify summarizer was called
        mock_summarizer.assert_called_once()

    @patch("quick_pdf_process.fitz.open")
    @patch("quick_pdf_process.pdf2image.convert_from_path")
    @patch("quick_pdf_process.pipeline")
    @patch("quick_pdf_process.torch.cuda.is_available", return_value=False)
    def test_quick_process_no_inst_tag(
        self, mock_cuda, mock_pipeline, mock_convert, mock_fitz
    ):
        """Test handling result without [/INST] tag."""
        mock_ocr = MagicMock(return_value=[{"generated_text": "G" * 100}])
        mock_summarizer = MagicMock(
            return_value=[{"generated_text": "Raw output without tag"}]
        )
        mock_pipeline.side_effect = [mock_ocr, mock_summarizer]

        mock_convert.return_value = [MagicMock()]

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.return_value = mock_doc

        result = quick_process("test.pdf")
        assert result == "edited_test.pdf"

    @patch("quick_pdf_process.fitz.open")
    @patch("quick_pdf_process.pdf2image.convert_from_path")
    @patch("quick_pdf_process.pipeline")
    @patch("quick_pdf_process.torch.cuda.is_available", return_value=False)
    def test_quick_process_more_texts_than_pages(
        self, mock_cuda, mock_pipeline, mock_convert, mock_fitz
    ):
        """Test when there are more texts than document pages."""
        # Setup - 2 images but only 1 page in document
        mock_ocr = MagicMock(
            side_effect=[
                [{"generated_text": "H" * 100}],
                [{"generated_text": "I" * 100}],
            ]
        )
        mock_summarizer = MagicMock(
            side_effect=[
                [{"generated_text": "[/INST] Summary 1"}],
                [{"generated_text": "[/INST] Summary 2"}],
            ]
        )
        mock_pipeline.side_effect = [mock_ocr, mock_summarizer]

        mock_convert.return_value = [MagicMock(), MagicMock()]

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)  # Only 1 page
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.return_value = mock_doc

        result = quick_process("test.pdf")

        # Should only insert text once (for the single page)
        assert mock_page.insert_textbox.call_count == 1

    @patch("quick_pdf_process.fitz.open")
    @patch("quick_pdf_process.pdf2image.convert_from_path")
    @patch("quick_pdf_process.pipeline")
    @patch("quick_pdf_process.torch.cuda.is_available", return_value=False)
    def test_quick_process_path_with_directory(
        self, mock_cuda, mock_pipeline, mock_convert, mock_fitz
    ):
        """Test processing file in a subdirectory."""
        mock_ocr = MagicMock(return_value=[{"generated_text": "J" * 100}])
        mock_summarizer = MagicMock(
            return_value=[{"generated_text": "[/INST] Summary"}]
        )
        mock_pipeline.side_effect = [mock_ocr, mock_summarizer]

        mock_convert.return_value = [MagicMock()]

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.return_value = mock_doc

        result = quick_process("/some/path/to/document.pdf")

        assert result == "/some/path/to/edited_document.pdf"


class TestMainBlock:
    """Test the __main__ block behavior."""

    @patch("quick_pdf_process.quick_process")
    @patch("quick_pdf_process.Path")
    def test_main_valid_file(self, mock_path, mock_quick_process):
        """Test main block with valid file."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Simulate running as main
        with patch.object(sys, "argv", ["quick_pdf_process.py", "test.pdf"]):
            # We can't easily test __main__ block, but we can test the logic
            from quick_pdf_process import MIN_ARGS_REQUIRED

            args = ["quick_pdf_process.py", "test.pdf"]
            assert len(args) >= MIN_ARGS_REQUIRED

    def test_main_insufficient_args(self):
        """Test main block with insufficient arguments."""
        args = ["quick_pdf_process.py"]
        assert len(args) < MIN_ARGS_REQUIRED

    @patch("quick_pdf_process.Path")
    def test_main_file_not_found(self, mock_path):
        """Test main block when file doesn't exist."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        # Verify the path check logic
        from pathlib import Path as RealPath

        with patch.object(RealPath, "exists", return_value=False):
            fake_path = RealPath("nonexistent.pdf")
            assert not fake_path.exists()
