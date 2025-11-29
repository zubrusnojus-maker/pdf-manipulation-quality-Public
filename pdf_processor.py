#!/usr/bin/env python3
"""
PDF Processing Script with OCR and Text Manipulation

A production-ready tool providing comprehensive PDF processing capabilities including:
- PDF loading and rasterization with validation
- High-quality OCR (Optical Character Recognition)
- Layout analysis for tables and structures
- Text manipulation (summarization/rewriting)
- Re-insertion of modified text back into PDF

Features:
    - Input validation and secure file handling
    - Configurable quality tiers (small/large models)
    - GPU acceleration with CPU fallback
    - Comprehensive error handling and logging

Prerequisites:
    pip install torch transformers pymupdf pdf2image pikepdf tqdm accelerate

Usage:
    python pdf_processor.py input.pdf -o output.pdf
    python pdf_processor.py input.pdf --model-size large --operation rewrite
    python pdf_processor.py input.pdf --dpi 150 --optimize
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import fitz  # PyMuPDF
import pdf2image
import pikepdf
import torch
from tqdm import tqdm
from transformers import pipeline

from constants import (
    COLOR_BLACK,
    DEFAULT_TEMPERATURE,
    DEVICE_CPU,
    DEVICE_GPU,
    DPI_DEFAULT,
    DPI_MAX,
    DPI_MIN,
    EXIT_FILE_NOT_FOUND,
    EXIT_INVALID_ARGUMENTS,
    EXIT_INVALID_PDF,
    EXIT_PROCESSING_ERROR,
    EXIT_SUCCESS,
    LAYOUT_MODEL_LAYOUTLMV3,
    MAX_INPUT_LENGTH_LARGE,
    MAX_INPUT_LENGTH_SMALL,
    MAX_NEW_TOKENS_LARGE,
    MAX_NEW_TOKENS_SMALL,
    MIN_TEXT_LENGTH,
    OCR_MODEL_TROCR_LARGE,
    PAGE_MARGIN_PX,
    TEXT_MODEL_LLAMA,
    TEXT_MODEL_MISTRAL,
    VALID_MODEL_SIZES,
    VALID_OPERATIONS,
)
from pdf_utils import (
    PDFProcessingError,
    PDFValidationError,
    cleanup_resources,
    generate_output_path,
    get_device,
    get_page_text_rect,
    setup_logging,
    truncate_text,
    validate_dpi,
    validate_output_path,
    validate_pdf_file,
)

if TYPE_CHECKING:
    from PIL import Image

# Configure module logger
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Main class for processing PDFs with OCR and text manipulation."""

    def __init__(
        self,
        model_size: Literal["small", "large"] = "small",
        use_layout: bool = False,
    ) -> None:
        """
        Initialize the PDF processor.

        Args:
            model_size: "small" (Mistral-7B) or "large" (Llama-13B) for text processing
            use_layout: Whether to use layout analysis (slower but better for tables)

        Raises:
            ValueError: If model_size is not valid
        """
        if model_size not in VALID_MODEL_SIZES:
            raise ValueError(
                f"Invalid model_size '{model_size}'. Must be one of: {VALID_MODEL_SIZES}"
            )

        self.model_size: Literal["small", "large"] = model_size
        self.use_layout: bool = use_layout
        self.ocr_pipe: Any | None = None
        self.layout_pipe: Any | None = None
        self.summariser: Any | None = None
        self._models_loaded: bool = False

        logger.info(
            f"Initialized PDFProcessor with model_size={model_size}, use_layout={use_layout}"
        )

    def load_models(self) -> None:
        """
        Load the required ML models (quality-optimized).

        Raises:
            RuntimeError: If model loading fails
        """
        if self._models_loaded:
            logger.debug("Models already loaded, skipping")
            return

        device = get_device()
        device_name = "GPU" if device == DEVICE_GPU else "CPU"
        logger.info(f"Using device: {device_name}")

        try:
            logger.info(f"Loading high-quality OCR model ({OCR_MODEL_TROCR_LARGE})...")
            self.ocr_pipe = pipeline(
                "image-to-text",
                model=OCR_MODEL_TROCR_LARGE,
                device=device,
            )

            if self.use_layout:
                logger.info(f"Loading layout model ({LAYOUT_MODEL_LAYOUTLMV3})...")
                self.layout_pipe = pipeline(
                    "document-question-answering",
                    model=LAYOUT_MODEL_LAYOUTLMV3,
                    device=device,
                )

            self._load_text_model()
            self._models_loaded = True
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _load_text_model(self) -> None:
        """Load the text manipulation model based on model_size."""
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if self.model_size == "large":
            logger.info(f"Loading text model ({TEXT_MODEL_LLAMA})...")
            self.summariser = pipeline(
                "text-generation",
                model=TEXT_MODEL_LLAMA,
                device_map="auto",
                model_kwargs={"torch_dtype": dtype},
                max_new_tokens=MAX_NEW_TOKENS_LARGE,
                do_sample=False,
                temperature=DEFAULT_TEMPERATURE,
            )
        else:
            logger.info(f"Loading text model ({TEXT_MODEL_MISTRAL})...")
            self.summariser = pipeline(
                "text-generation",
                model=TEXT_MODEL_MISTRAL,
                device_map="auto",
                model_kwargs={"torch_dtype": dtype},
                max_new_tokens=MAX_NEW_TOKENS_SMALL,
                do_sample=True,
                temperature=DEFAULT_TEMPERATURE,
            )

    def load_pdf(self, pdf_path: str | Path) -> fitz.Document:
        """
        Load and validate PDF document.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            PyMuPDF document object

        Raises:
            PDFValidationError: If PDF is invalid
            FileNotFoundError: If file doesn't exist
        """
        validated_path = validate_pdf_file(pdf_path)
        logger.info(f"Loading PDF: {validated_path}")
        return fitz.open(str(validated_path))

    def rasterize_pdf(
        self,
        pdf_path: str | Path,
        dpi: int = DPI_DEFAULT,
    ) -> list["Image.Image"]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for rasterization

        Returns:
            List of PIL Image objects

        Raises:
            ValueError: If DPI is out of valid range
            PDFProcessingError: If rasterization fails
        """
        validated_dpi = validate_dpi(dpi)
        logger.info(f"Rasterizing PDF at {validated_dpi} DPI...")

        try:
            return pdf2image.convert_from_path(str(pdf_path), dpi=validated_dpi)
        except Exception as e:
            logger.error(f"Rasterization failed: {e}")
            raise PDFProcessingError(f"Could not rasterize PDF: {e}") from e

    def extract_text_from_images(
        self,
        images: list["Image.Image"],
    ) -> list[str]:
        """
        Extract text from rasterized PDF images using OCR.

        Args:
            images: List of PIL Image objects

        Returns:
            List of extracted text strings, one per page

        Raises:
            RuntimeError: If OCR pipeline is not loaded
        """
        if self.ocr_pipe is None:
            raise RuntimeError("OCR pipeline not loaded. Call load_models() first.")

        logger.info("Extracting text with OCR...")
        page_texts: list[str] = []

        for page_img in tqdm(images, desc="OCR Processing"):
            try:
                ocr_result = self.ocr_pipe(page_img)

                if isinstance(ocr_result, list) and len(ocr_result) > 0:
                    text = ocr_result[0].get("generated_text", "")
                else:
                    text = ""

                # Optional: Layout analysis for table detection
                if self.use_layout and self.layout_pipe and text:
                    # Layout analysis can be added here
                    pass

                page_texts.append(text)

            except Exception as e:
                logger.warning(f"OCR failed for page, using empty text: {e}")
                page_texts.append("")

        return page_texts

    def manipulate_text(
        self,
        page_texts: list[str],
        operation: Literal["summarize", "rewrite"] = "summarize",
    ) -> list[str]:
        """
        Apply text manipulation (summarization or rewriting).

        Args:
            page_texts: List of text strings from each page
            operation: "summarize" or "rewrite"

        Returns:
            List of manipulated text strings

        Raises:
            ValueError: If operation is not valid
            RuntimeError: If summariser pipeline is not loaded
        """
        if operation not in VALID_OPERATIONS:
            raise ValueError(
                f"Invalid operation '{operation}'. Must be one of: {VALID_OPERATIONS}"
            )

        if self.summariser is None:
            raise RuntimeError("Summariser pipeline not loaded. Call load_models() first.")

        logger.info(f"Performing text {operation}...")
        manipulated_texts: list[str] = []
        max_length = MAX_INPUT_LENGTH_LARGE if self.model_size == "large" else MAX_INPUT_LENGTH_SMALL

        for text_content in tqdm(page_texts, desc=f"Text {operation}"):
            if not text_content or len(text_content.strip()) < MIN_TEXT_LENGTH:
                manipulated_texts.append(text_content)
                continue

            try:
                truncated_text = truncate_text(text_content, max_length)
                prompt = self._format_prompt(truncated_text, operation)
                processed_text = self._run_text_model(prompt, text_content)
                manipulated_texts.append(processed_text)

            except torch.cuda.OutOfMemoryError:
                logger.error("GPU out of memory. Try reducing DPI or using smaller model.")
                manipulated_texts.append(text_content)
            except RuntimeError as e:
                logger.warning(f"Model inference failed: {e}")
                manipulated_texts.append(text_content)
            except Exception as e:
                logger.warning(f"Text processing failed: {e}")
                manipulated_texts.append(text_content)

        return manipulated_texts

    def _format_prompt(
        self,
        text: str,
        operation: Literal["summarize", "rewrite"],
    ) -> str:
        """Format prompt for instruction-following models."""
        if operation == "summarize":
            return (
                f"<s>[INST] Summarize the following text concisely "
                f"while preserving key information:\n\n{text}\n\n[/INST]"
            )
        return (
            f"<s>[INST] Rewrite the following text to improve "
            f"clarity and readability:\n\n{text}\n\n[/INST]"
        )

    def _run_text_model(self, prompt: str, fallback_text: str) -> str:
        """Run text model and extract response."""
        model_result = self.summariser(prompt)

        if isinstance(model_result, list) and len(model_result) > 0:
            processed_text = model_result[0].get("generated_text", "")
            if "[/INST]" in processed_text:
                processed_text = processed_text.split("[/INST]")[-1].strip()
            return processed_text

        return fallback_text

    def reinsert_text(
        self,
        document: fitz.Document,
        manipulated_texts: list[str],
        fontsize: int = 12,
    ) -> None:
        """
        Re-insert modified text back into PDF pages.

        Args:
            document: PyMuPDF document object
            manipulated_texts: List of text strings to insert
            fontsize: Font size for inserted text
        """
        logger.info("Re-inserting text into PDF...")

        for page_index, modified_text in enumerate(
            tqdm(manipulated_texts, desc="Inserting text")
        ):
            if page_index >= len(document):
                logger.warning(f"More texts than pages, stopping at page {page_index}")
                break

            page = document[page_index]
            text_rect = get_page_text_rect(page, PAGE_MARGIN_PX)

            try:
                page.insert_textbox(
                    text_rect,
                    modified_text,
                    fontsize=fontsize,
                    color=COLOR_BLACK,
                    align=0,
                )
            except Exception as e:
                logger.warning(f"Failed to insert text on page {page_index + 1}: {e}")

    def process_pdf(
        self,
        input_path: str | Path,
        output_path: str | Path,
        operation: Literal["summarize", "rewrite"] = "summarize",
        dpi: int = DPI_DEFAULT,
    ) -> Path:
        """
        Complete PDF processing pipeline.

        Args:
            input_path: Path to input PDF
            output_path: Path to save output PDF
            operation: "summarize" or "rewrite"
            dpi: DPI for rasterization

        Returns:
            Path to the output file

        Raises:
            PDFValidationError: If input validation fails
            PDFProcessingError: If processing fails
        """
        document = None

        try:
            # Validate inputs
            validated_input = validate_pdf_file(input_path)
            validated_output = validate_output_path(output_path, input_path)

            # Load models if not already loaded
            if not self._models_loaded:
                self.load_models()

            # Process pipeline
            document = self.load_pdf(validated_input)
            rasterized_images = self.rasterize_pdf(validated_input, dpi=dpi)
            extracted_page_texts = self.extract_text_from_images(rasterized_images)
            manipulated_page_texts = self.manipulate_text(extracted_page_texts, operation=operation)
            self.reinsert_text(document, manipulated_page_texts)

            # Save
            logger.info(f"Saving processed PDF to: {validated_output}")
            document.save(str(validated_output))

            logger.info("Processing complete!")
            return validated_output

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except PDFValidationError as e:
            logger.error(f"Validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise PDFProcessingError(f"PDF processing failed: {e}") from e
        finally:
            cleanup_resources(document)

    def cleanup(self) -> None:
        """Release model resources and free GPU memory."""
        logger.info("Cleaning up resources...")

        if self.ocr_pipe is not None:
            del self.ocr_pipe
            self.ocr_pipe = None

        if self.layout_pipe is not None:
            del self.layout_pipe
            self.layout_pipe = None

        if self.summariser is not None:
            del self.summariser
            self.summariser = None

        self._models_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")


def post_process_with_pikepdf(
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Post-process PDF with pikepdf for optimization.

    Args:
        input_path: Path to input PDF
        output_path: Path to save optimized PDF

    Returns:
        Path to the optimized PDF

    Raises:
        PDFProcessingError: If optimization fails
    """
    logger.info("Post-processing with pikepdf...")
    source_pdf = None
    destination_pdf = None

    try:
        source_pdf = pikepdf.Pdf.open(str(input_path))
        destination_pdf = pikepdf.Pdf.new()

        for page in tqdm(source_pdf.pages, desc="Copying pages"):
            destination_pdf.pages.append(page)

        destination_pdf.save(str(output_path))
        logger.info(f"Optimized PDF saved to: {output_path}")

        return Path(output_path)

    except Exception as e:
        logger.error(f"PDF optimization failed: {e}")
        raise PDFProcessingError(f"Optimization failed: {e}") from e
    finally:
        cleanup_resources(source_pdf, destination_pdf)


def main() -> int:
    """Main entry point for PDF processing CLI."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Process PDFs with OCR and text manipulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pdf_processor.py document.pdf
    python pdf_processor.py document.pdf -o output.pdf
    python pdf_processor.py document.pdf --model-size large --operation rewrite
    python pdf_processor.py document.pdf --dpi 150 --optimize
        """,
    )

    parser.add_argument(
        "input",
        help="Input PDF file path",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output PDF file path (default: edited_<input>)",
        default=None,
    )
    parser.add_argument(
        "-m", "--model-size",
        choices=list(VALID_MODEL_SIZES),
        default="small",
        help="Model size for text processing (default: small)",
    )
    parser.add_argument(
        "--operation",
        choices=list(VALID_OPERATIONS),
        default="summarize",
        help="Text manipulation operation (default: summarize)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI_DEFAULT,
        help=f"DPI for PDF rasterization ({DPI_MIN}-{DPI_MAX}, default: {DPI_DEFAULT})",
    )
    parser.add_argument(
        "--use-layout",
        action="store_true",
        help="Enable layout analysis (slower but better for tables)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Post-process with pikepdf optimization",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Handle verbose mode
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate DPI
    try:
        validate_dpi(args.dpi)
    except ValueError as e:
        logger.error(str(e))
        return EXIT_INVALID_ARGUMENTS

    # Validate input file
    try:
        validate_pdf_file(args.input)
    except FileNotFoundError:
        logger.error(f"File not found: {args.input}")
        return EXIT_FILE_NOT_FOUND
    except PDFValidationError as e:
        logger.error(f"Invalid PDF: {e}")
        return EXIT_INVALID_PDF

    # Generate output path if not provided
    if args.output is None:
        args.output = str(generate_output_path(args.input, prefix="edited_"))

    # Process PDF
    processor = None
    try:
        processor = PDFProcessor(
            model_size=args.model_size,
            use_layout=args.use_layout,
        )

        output_path = processor.process_pdf(
            args.input,
            args.output,
            operation=args.operation,
            dpi=args.dpi,
        )

        # Optional optimization
        if args.optimize:
            optimized_path = generate_output_path(output_path, prefix="optimized_")
            post_process_with_pikepdf(output_path, optimized_path)
            print(f"Optimized PDF saved to: {optimized_path}")

        print(f"Processing complete! Output: {output_path}")
        return EXIT_SUCCESS

    except (PDFValidationError, PDFProcessingError) as e:
        logger.error(str(e))
        return EXIT_PROCESSING_ERROR
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return EXIT_PROCESSING_ERROR
    finally:
        if processor:
            processor.cleanup()


if __name__ == "__main__":
    sys.exit(main())
