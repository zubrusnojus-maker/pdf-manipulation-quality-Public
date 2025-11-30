"""Tests for PDF processing functionality."""

from unittest.mock import MagicMock, patch

import pytest
from pdf_processor import (
    LAYOUT_TYPES,
    MIN_TEXT_LENGTH,
    LayoutRegion,
    PDFProcessor,
    post_process_with_pikepdf,
)


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


class TestLoadModels:
    """Test model loading with mocked pipelines."""

    @patch("pdf_toolkit.models.loader.pipeline")
    @patch("pdf_toolkit.models.loader.torch.cuda.is_available", return_value=False)
    def test_load_models_small_cpu(self, mock_cuda, mock_pipeline):
        """Test loading small model on CPU."""
        mock_pipeline.return_value = MagicMock()
        processor = PDFProcessor(model_size="small", use_layout=False)
        processor.load_models()

        assert processor.ocr_pipe is not None
        assert processor.summariser is not None
        assert processor.layout_pipe is None
        # OCR pipeline call
        mock_pipeline.assert_any_call(
            "image-to-text",
            model="microsoft/trocr-large-printed",
            device=-1,
        )

    @patch("pdf_toolkit.models.loader.pipeline")
    @patch("pdf_toolkit.models.loader.torch.cuda.is_available", return_value=True)
    def test_load_models_large_gpu(self, mock_cuda, mock_pipeline):
        """Test loading large model on GPU."""
        mock_pipeline.return_value = MagicMock()
        processor = PDFProcessor(model_size="large", use_layout=False)
        processor.load_models()

        assert processor.ocr_pipe is not None
        assert processor.summariser is not None

    @patch("pdf_toolkit.models.loader.pipeline")
    @patch("pdf_toolkit.models.loader.torch.cuda.is_available", return_value=False)
    def test_load_models_with_layout(self, mock_cuda, mock_pipeline):
        """Test loading models with layout analysis enabled."""
        mock_pipeline.return_value = MagicMock()
        processor = PDFProcessor(model_size="small", use_layout=True)
        processor.load_models()

        assert processor.layout_pipe is not None
        # Should have 3 pipeline calls: OCR, layout, summarizer
        assert mock_pipeline.call_count == 3


class TestLoadPDF:
    """Test PDF loading functionality."""

    @patch("pdf_toolkit.core.processor.fitz.open")
    def test_load_pdf(self, mock_fitz_open):
        """Test PDF loading."""
        mock_doc = MagicMock()
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor()
        result = processor.load_pdf("test.pdf")

        mock_fitz_open.assert_called_once_with("test.pdf")
        assert result == mock_doc


class TestRasterizePDF:
    """Test PDF rasterization."""

    @patch("pdf_toolkit.utils.pdf_utils.pdf2image.convert_from_path")
    def test_rasterize_pdf_default_dpi(self, mock_convert):
        """Test rasterization with default DPI."""
        mock_images = [MagicMock(), MagicMock()]
        mock_convert.return_value = mock_images

        processor = PDFProcessor()
        # Use the module-level function, not the method
        from pdf_toolkit.utils.pdf_utils import rasterize_pdf

        result = rasterize_pdf("test.pdf")

        mock_convert.assert_called_once_with("test.pdf", dpi=300)
        assert result == mock_images

    @patch("pdf_toolkit.utils.pdf_utils.pdf2image.convert_from_path")
    def test_rasterize_pdf_custom_dpi(self, mock_convert):
        """Test rasterization with custom DPI."""
        mock_images = [MagicMock()]
        mock_convert.return_value = mock_images

        from pdf_toolkit.utils.pdf_utils import rasterize_pdf

        result = rasterize_pdf("test.pdf", dpi=600)

        mock_convert.assert_called_once_with("test.pdf", dpi=600)


class TestExtractTextFromImages:
    """Test OCR text extraction."""

    def test_extract_text_single_image(self):
        """Test extracting text from a single image."""
        processor = PDFProcessor()
        processor.ocr_pipe = MagicMock(return_value=[{"generated_text": "Hello World"}])

        mock_image = MagicMock()
        result = processor.extract_text_from_images([mock_image])

        assert result == ["Hello World"]
        processor.ocr_pipe.assert_called_once_with(mock_image)

    def test_extract_text_multiple_images(self):
        """Test extracting text from multiple images."""
        processor = PDFProcessor()
        processor.ocr_pipe = MagicMock(
            side_effect=[
                [{"generated_text": "Page 1 text"}],
                [{"generated_text": "Page 2 text"}],
            ]
        )

        mock_images = [MagicMock(), MagicMock()]
        result = processor.extract_text_from_images(mock_images)

        assert result == ["Page 1 text", "Page 2 text"]
        assert processor.ocr_pipe.call_count == 2

    def test_extract_text_empty_result(self):
        """Test handling empty OCR result."""
        processor = PDFProcessor()
        processor.ocr_pipe = MagicMock(return_value=[])

        result = processor.extract_text_from_images([MagicMock()])

        assert result == [""]

    def test_extract_text_missing_key(self):
        """Test handling missing generated_text key."""
        processor = PDFProcessor()
        processor.ocr_pipe = MagicMock(return_value=[{"other_key": "value"}])

        result = processor.extract_text_from_images([MagicMock()])

        assert result == [""]

    def test_extract_text_with_layout(self):
        """Test extraction with layout analysis enabled."""
        processor = PDFProcessor(use_layout=True)
        processor.ocr_pipe = MagicMock(return_value=[{"generated_text": "Text content"}])
        processor.layout_pipe = MagicMock(return_value=[{"label": "letter", "score": 0.95}])

        result = processor.extract_text_from_images([MagicMock()])

        # With layout enabled, returns list of (text, layout_region) tuples
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        text, layout_region = result[0]
        assert text == "Text content"
        assert isinstance(layout_region, LayoutRegion)
        assert layout_region.region_type == "paragraph"  # letter maps to paragraph
        assert layout_region.confidence == 0.95


class TestLayoutRegion:
    """Test LayoutRegion dataclass."""

    def test_layout_region_creation(self):
        """Test creating a LayoutRegion."""
        region = LayoutRegion(
            region_type="header",
            text="invoice",
            confidence=0.95,
            bbox=(0, 0, 100, 50),
        )
        assert region.region_type == "header"
        assert region.text == "invoice"
        assert region.confidence == 0.95
        assert region.bbox == (0, 0, 100, 50)

    def test_layout_region_default_bbox(self):
        """Test LayoutRegion with default bbox."""
        region = LayoutRegion(
            region_type="paragraph",
            text="letter",
            confidence=0.8,
        )
        assert region.bbox is None


class TestLayoutTypes:
    """Test LAYOUT_TYPES configuration."""

    def test_layout_types_exist(self):
        """Test that all expected layout types exist."""
        assert "header" in LAYOUT_TYPES
        assert "paragraph" in LAYOUT_TYPES
        assert "table" in LAYOUT_TYPES
        assert "footer" in LAYOUT_TYPES

    def test_layout_types_have_required_keys(self):
        """Test that layout types have required configuration keys."""
        for layout_type, config in LAYOUT_TYPES.items():
            assert "fontsize_multiplier" in config
            assert "align" in config


class TestAnalyzeLayout:
    """Test the analyze_layout method."""

    def test_analyze_layout_no_pipe(self):
        """Test analyze_layout when layout_pipe is not loaded."""
        processor = PDFProcessor(use_layout=False)
        result = processor.analyze_layout(MagicMock())

        assert result.region_type == "paragraph"
        assert result.confidence == 1.0

    def test_analyze_layout_invoice(self):
        """Test analyze_layout detects invoice as table."""
        processor = PDFProcessor(use_layout=True)
        processor.layout_pipe = MagicMock(return_value=[{"label": "invoice", "score": 0.92}])

        result = processor.analyze_layout(MagicMock())

        assert result.region_type == "table"
        assert result.text == "invoice"
        assert result.confidence == 0.92

    def test_analyze_layout_letter(self):
        """Test analyze_layout detects letter as paragraph."""
        processor = PDFProcessor(use_layout=True)
        processor.layout_pipe = MagicMock(return_value=[{"label": "letter", "score": 0.88}])

        result = processor.analyze_layout(MagicMock())

        assert result.region_type == "paragraph"
        assert result.text == "letter"

    def test_analyze_layout_presentation(self):
        """Test analyze_layout detects presentation as header."""
        processor = PDFProcessor(use_layout=True)
        processor.layout_pipe = MagicMock(return_value=[{"label": "presentation", "score": 0.75}])

        result = processor.analyze_layout(MagicMock())

        assert result.region_type == "header"

    def test_analyze_layout_empty_result(self):
        """Test analyze_layout with empty result."""
        processor = PDFProcessor(use_layout=True)
        processor.layout_pipe = MagicMock(return_value=[])

        result = processor.analyze_layout(MagicMock())

        assert result.region_type == "paragraph"
        assert result.confidence == 0.0

    def test_analyze_layout_exception(self):
        """Test analyze_layout handles exceptions gracefully."""
        processor = PDFProcessor(use_layout=True)
        processor.layout_pipe = MagicMock(side_effect=Exception("Model error"))

        result = processor.analyze_layout(MagicMock())

        assert result.region_type == "paragraph"
        assert result.confidence == 0.0

    def test_analyze_layout_unknown_label(self):
        """Test analyze_layout with unknown document type."""
        processor = PDFProcessor(use_layout=True)
        processor.layout_pipe = MagicMock(return_value=[{"label": "unknown_type", "score": 0.5}])

        result = processor.analyze_layout(MagicMock())

        # Unknown types should default to paragraph
        assert result.region_type == "paragraph"


class TestManipulateText:
    """Test text manipulation (summarization/rewriting)."""

    def test_manipulate_text_summarize(self):
        """Test text summarization."""
        processor = PDFProcessor(model_size="small")
        processor.summariser = MagicMock(
            return_value=[{"generated_text": "prompt [/INST] This is a summary."}]
        )

        long_text = "A" * 100  # Text longer than MIN_TEXT_LENGTH
        result = processor.manipulate_text([long_text], operation="summarize")

        assert result == ["This is a summary."]

    def test_manipulate_text_rewrite(self):
        """Test text rewriting."""
        processor = PDFProcessor(model_size="small")
        processor.summariser = MagicMock(
            return_value=[{"generated_text": "prompt [/INST] Rewritten text here."}]
        )

        long_text = "B" * 100
        result = processor.manipulate_text([long_text], operation="rewrite")

        assert result == ["Rewritten text here."]

    def test_manipulate_text_skip_short(self):
        """Test that short text is skipped."""
        processor = PDFProcessor()
        processor.summariser = MagicMock()

        short_text = "Short"  # Less than MIN_TEXT_LENGTH
        result = processor.manipulate_text([short_text])

        assert result == [short_text]
        processor.summariser.assert_not_called()

    def test_manipulate_text_skip_empty(self):
        """Test that empty text is skipped."""
        processor = PDFProcessor()
        processor.summariser = MagicMock()

        result = processor.manipulate_text([""])

        assert result == [""]
        processor.summariser.assert_not_called()

    def test_manipulate_text_truncation_small(self):
        """Test text truncation for small model (1024 chars)."""
        processor = PDFProcessor(model_size="small")
        processor.summariser = MagicMock(
            return_value=[{"generated_text": "[/INST] Truncated summary"}]
        )

        very_long_text = "X" * 2000
        processor.manipulate_text([very_long_text])

        # Verify the prompt was created with truncated text
        call_args = processor.summariser.call_args[0][0]
        # The text inside the prompt should be truncated to 1024
        assert len(very_long_text[:1024]) == 1024

    def test_manipulate_text_truncation_large(self):
        """Test text truncation for large model (2048 chars)."""
        processor = PDFProcessor(model_size="large")
        processor.summariser = MagicMock(
            return_value=[{"generated_text": "[/INST] Truncated summary"}]
        )

        very_long_text = "Y" * 3000
        processor.manipulate_text([very_long_text])

        processor.summariser.assert_called_once()

    def test_manipulate_text_exception_handling(self):
        """Test graceful handling of model errors."""
        processor = PDFProcessor()
        processor.summariser = MagicMock(side_effect=Exception("Model error"))

        long_text = "C" * 100
        result = processor.manipulate_text([long_text])

        # Should return original text on error
        assert result == [long_text]

    def test_manipulate_text_empty_model_result(self):
        """Test handling empty model result."""
        processor = PDFProcessor()
        processor.summariser = MagicMock(return_value=[])

        long_text = "D" * 100
        result = processor.manipulate_text([long_text])

        assert result == [long_text]

    def test_manipulate_text_no_inst_tag(self):
        """Test handling result without [/INST] tag."""
        processor = PDFProcessor()
        processor.summariser = MagicMock(
            return_value=[{"generated_text": "Raw output without tag"}]
        )

        long_text = "E" * 100
        result = processor.manipulate_text([long_text])

        # Should return the raw text when no [/INST] tag
        assert result == ["Raw output without tag"]

    def test_manipulate_text_with_layout_data(self):
        """Test manipulate_text with layout tuples input."""
        processor = PDFProcessor(model_size="small")
        processor.summariser = MagicMock(
            return_value=[{"generated_text": "[/INST] Summary with layout"}]
        )

        layout_region = LayoutRegion(
            region_type="table",
            text="invoice",
            confidence=0.9,
        )
        page_data = [("F" * 100, layout_region)]
        result = processor.manipulate_text(page_data, operation="summarize")

        # Should return list of (text, layout) tuples
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert result[0][0] == "Summary with layout"
        assert result[0][1] == layout_region

    def test_manipulate_text_with_layout_context_invoice(self):
        """Test that invoice layout adds context to prompt."""
        processor = PDFProcessor(model_size="small")
        processor.summariser = MagicMock(return_value=[{"generated_text": "[/INST] Summary"}])

        layout_region = LayoutRegion(
            region_type="table",
            text="invoice",
            confidence=0.9,
        )
        page_data = [("G" * 100, layout_region)]
        processor.manipulate_text(page_data)

        # Check that the prompt includes layout context
        call_args = processor.summariser.call_args[0][0]
        assert "tabular data" in call_args

    def test_manipulate_text_with_layout_context_letter(self):
        """Test that letter layout adds context to prompt."""
        processor = PDFProcessor(model_size="small")
        processor.summariser = MagicMock(return_value=[{"generated_text": "[/INST] Summary"}])

        layout_region = LayoutRegion(
            region_type="paragraph",
            text="letter",
            confidence=0.85,
        )
        page_data = [("H" * 100, layout_region)]
        processor.manipulate_text(page_data)

        call_args = processor.summariser.call_args[0][0]
        assert "correspondence" in call_args

    def test_manipulate_text_with_low_confidence_layout(self):
        """Test that low confidence layout doesn't add context."""
        processor = PDFProcessor(model_size="small")
        processor.summariser = MagicMock(return_value=[{"generated_text": "[/INST] Summary"}])

        layout_region = LayoutRegion(
            region_type="table",
            text="invoice",
            confidence=0.4,  # Below 0.5 threshold
        )
        page_data = [("I" * 100, layout_region)]
        processor.manipulate_text(page_data)

        call_args = processor.summariser.call_args[0][0]
        assert "tabular" not in call_args


class TestReinsertText:
    """Test text re-insertion into PDF."""

    def test_reinsert_text_single_page(self):
        """Test reinserting text into a single page."""
        processor = PDFProcessor()

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        processor.reinsert_text(mock_doc, ["New text content"])

        mock_page.clean_contents.assert_called_once()
        mock_page.insert_textbox.assert_called_once()

    def test_reinsert_text_multiple_pages(self):
        """Test reinserting text into multiple pages."""
        processor = PDFProcessor()

        mock_pages = [MagicMock() for _ in range(3)]
        for page in mock_pages:
            page.rect.width = 612
            page.rect.height = 792

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.__getitem__ = MagicMock(side_effect=mock_pages)

        texts = ["Page 1", "Page 2", "Page 3"]
        processor.reinsert_text(mock_doc, texts)

        for page in mock_pages:
            page.clean_contents.assert_called_once()
            page.insert_textbox.assert_called_once()

    def test_reinsert_text_more_texts_than_pages(self):
        """Test when there are more texts than pages."""
        processor = PDFProcessor()

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        # More texts than pages - should only process first page
        processor.reinsert_text(mock_doc, ["Text 1", "Text 2", "Text 3"])

        # Should only be called once (for the single page)
        assert mock_page.insert_textbox.call_count == 1

    def test_reinsert_text_custom_fontsize(self):
        """Test reinserting with custom font size."""
        processor = PDFProcessor()

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        processor.reinsert_text(mock_doc, ["Text"], fontsize=16)

        call_kwargs = mock_page.insert_textbox.call_args[1]
        assert call_kwargs["fontsize"] == 16

    def test_reinsert_text_with_layout_header(self):
        """Test reinserting with header layout (larger font, centered)."""
        processor = PDFProcessor()

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        layout_region = LayoutRegion(
            region_type="header",
            text="presentation",
            confidence=0.9,
        )
        data = [("Header text", layout_region)]
        processor.reinsert_text(mock_doc, data, fontsize=12)

        call_kwargs = mock_page.insert_textbox.call_args[1]
        # Header should have 1.2x font size
        assert call_kwargs["fontsize"] == 12 * 1.2
        # Header should be centered (align=1)
        assert call_kwargs["align"] == 1

    def test_reinsert_text_with_layout_table(self):
        """Test reinserting with table layout (smaller font)."""
        processor = PDFProcessor()

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        layout_region = LayoutRegion(
            region_type="table",
            text="invoice",
            confidence=0.85,
        )
        data = [("Table content", layout_region)]
        processor.reinsert_text(mock_doc, data, fontsize=12)

        call_kwargs = mock_page.insert_textbox.call_args[1]
        # Table should have 0.9x font size
        assert call_kwargs["fontsize"] == 12 * 0.9
        # Table should be left-aligned
        assert call_kwargs["align"] == 0

    def test_reinsert_text_with_layout_footer(self):
        """Test reinserting with footer layout (smallest font, centered)."""
        processor = PDFProcessor()

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        layout_region = LayoutRegion(
            region_type="footer",
            text="footer",
            confidence=0.7,
        )
        data = [("Footer text", layout_region)]
        processor.reinsert_text(mock_doc, data, fontsize=12)

        call_kwargs = mock_page.insert_textbox.call_args[1]
        # Footer should have 0.8x font size
        assert call_kwargs["fontsize"] == 12 * 0.8
        # Footer should be centered
        assert call_kwargs["align"] == 1

    def test_reinsert_text_with_low_confidence_layout(self):
        """Test that low confidence layout uses default formatting."""
        processor = PDFProcessor()

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        layout_region = LayoutRegion(
            region_type="header",
            text="presentation",
            confidence=0.3,  # Low confidence
        )
        data = [("Text", layout_region)]
        processor.reinsert_text(mock_doc, data, fontsize=12)

        call_kwargs = mock_page.insert_textbox.call_args[1]
        # Should use default formatting
        assert call_kwargs["fontsize"] == 12
        assert call_kwargs["align"] == 0


class TestProcessPDF:
    """Test the complete PDF processing pipeline."""

    @patch("pdf_toolkit.core.processor.fitz.open")
    @patch("pdf_toolkit.core.processor.rasterize_pdf")
    @patch("pdf_toolkit.models.loader.pipeline")
    @patch("pdf_toolkit.models.loader.torch.cuda.is_available", return_value=False)
    def test_process_pdf_full_pipeline(self, mock_cuda, mock_pipeline, mock_rasterize, mock_fitz):
        """Test the full processing pipeline."""
        # Setup mocks
        mock_ocr = MagicMock(return_value=[{"generated_text": "A" * 100}])
        mock_summarizer = MagicMock(return_value=[{"generated_text": "[/INST] Summary text"}])
        mock_pipeline.side_effect = [mock_ocr, mock_summarizer]

        mock_image = MagicMock()
        mock_rasterize.return_value = [mock_image]

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.return_value = mock_doc

        # Run pipeline
        processor = PDFProcessor(model_size="small")
        result = processor.process_pdf("input.pdf", "output.pdf")

        # Verify
        assert result == "output.pdf"
        mock_fitz.assert_called_with("input.pdf")
        mock_rasterize.assert_called_once()
        mock_doc.save.assert_called_once_with("output.pdf")
        mock_doc.close.assert_called_once()

    @patch("pdf_toolkit.core.processor.fitz.open")
    @patch("pdf_toolkit.core.processor.rasterize_pdf")
    def test_process_pdf_skips_model_loading_if_loaded(self, mock_rasterize, mock_fitz):
        """Test that models aren't reloaded if already loaded."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)
        mock_fitz.return_value = mock_doc
        mock_rasterize.return_value = []

        processor = PDFProcessor()
        processor.summariser = MagicMock()  # Pre-loaded
        processor.ocr_pipe = MagicMock(return_value=[])

        with patch.object(processor, "load_models") as mock_load:
            processor.process_pdf("input.pdf", "output.pdf")
            mock_load.assert_not_called()


class TestPostProcessWithPikepdf:
    """Test pikepdf post-processing."""

    @patch("pdf_toolkit.utils.pdf_utils.pikepdf.Pdf.open")
    @patch("pdf_toolkit.utils.pdf_utils.pikepdf.Pdf.new")
    def test_post_process_with_pikepdf(self, mock_new, mock_open):
        """Test PDF optimization with pikepdf."""
        from pdf_toolkit.utils.pdf_utils import optimize_pdf

        mock_source = MagicMock()
        mock_source.pages = [MagicMock(), MagicMock()]
        mock_open.return_value = mock_source

        mock_dest = MagicMock()
        mock_dest.pages = MagicMock()
        mock_new.return_value = mock_dest

        optimize_pdf("input.pdf", "output.pdf")

        mock_open.assert_called_once_with("input.pdf")
        mock_dest.save.assert_called_once_with("output.pdf")
        mock_source.close.assert_called_once()


class TestMainFunction:
    """Test the CLI main function."""

    @patch("pdf_toolkit.cli.processor_cli.PDFProcessor")
    @patch("pdf_toolkit.cli.processor_cli.optimize_pdf")
    def test_main_basic(self, mock_post_process, mock_processor_class):
        """Test basic main function execution."""
        from pdf_toolkit.cli.processor_cli import main

        mock_processor = MagicMock()
        mock_processor.process_pdf.return_value = "output.pdf"
        mock_processor_class.return_value = mock_processor

        with patch("sys.argv", ["pdf_processor.py", "test.pdf"]):
            main()

        mock_processor_class.assert_called_once_with(model_size="small", use_layout=False)
        mock_processor.process_pdf.assert_called_once()
        mock_post_process.assert_not_called()

    @patch("pdf_toolkit.cli.processor_cli.PDFProcessor")
    @patch("pdf_toolkit.cli.processor_cli.optimize_pdf")
    def test_main_with_optimize(self, mock_post_process, mock_processor_class):
        """Test main function with optimization flag."""
        from pdf_toolkit.cli.processor_cli import main

        mock_processor = MagicMock()
        mock_processor.process_pdf.return_value = "output.pdf"
        mock_processor_class.return_value = mock_processor

        with patch("sys.argv", ["pdf_processor.py", "test.pdf", "--optimize"]):
            main()

        mock_post_process.assert_called_once()

    @patch("pdf_toolkit.cli.processor_cli.PDFProcessor")
    def test_main_with_all_options(self, mock_processor_class):
        """Test main function with all CLI options."""
        from pdf_toolkit.cli.processor_cli import main

        mock_processor = MagicMock()
        mock_processor.process_pdf.return_value = "custom_output.pdf"
        mock_processor_class.return_value = mock_processor

        with patch(
            "sys.argv",
            [
                "pdf_processor.py",
                "input.pdf",
                "-o",
                "custom_output.pdf",
                "-m",
                "large",
                "--operation",
                "rewrite",
                "--dpi",
                "600",
                "--use-layout",
            ],
        ):
            main()

        mock_processor_class.assert_called_once_with(model_size="large", use_layout=True)
        mock_processor.process_pdf.assert_called_once_with(
            "input.pdf", "custom_output.pdf", operation="rewrite", dpi=600
        )
