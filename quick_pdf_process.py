#!/usr/bin/env python3
"""
Quick-start one-liner script for PDF processing.

A simplified interface for quick PDF processing with sensible defaults.
Uses the same underlying functionality as pdf_processor.py but with
minimal configuration required.

Usage:
    python quick_pdf_process.py mydoc.pdf

This will create 'edited_mydoc.pdf' with summarized content.

For more options, use pdf_processor.py directly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import fitz
import pdf2image
import torch
from transformers import pipeline

from constants import (
    COLOR_BLACK,
    DPI_DEFAULT,
    EXIT_FILE_NOT_FOUND,
    EXIT_INVALID_PDF,
    EXIT_PROCESSING_ERROR,
    EXIT_SUCCESS,
    MAX_INPUT_LENGTH_SMALL,
    MIN_TEXT_LENGTH,
    OCR_MODEL_TROCR_LARGE,
    PAGE_MARGIN_PX,
    TEXT_MODEL_MISTRAL,
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
    validate_pdf_file,
)

if TYPE_CHECKING:
    from PIL import Image

# Configure module logger
logger = logging.getLogger(__name__)


def quick_process(pdf_path: str | Path) -> Path:
    """
    Quick PDF processing in one function.

    Provides a simplified interface for processing PDFs with:
    - High-quality OCR (TrOCR Large)
    - Text summarization (Mistral-7B)
    - Automatic output naming

    Args:
        pdf_path: Path to the input PDF file

    Returns:
        Path to the output PDF file

    Raises:
        PDFValidationError: If input file is invalid
        PDFProcessingError: If processing fails
        FileNotFoundError: If file doesn't exist
    """
    document = None

    try:
        # Validate input
        validated_path = validate_pdf_file(pdf_path)
        logger.info(f"Processing: {validated_path}")

        # Generate output path
        output_path = generate_output_path(validated_path, prefix="edited_")

        # Load PDF and rasterize
        logger.info("Loading and rasterizing PDF...")
        document = fitz.open(str(validated_path))
        images: list[Image.Image] = pdf2image.convert_from_path(
            str(validated_path), dpi=DPI_DEFAULT
        )

        # OCR
        device = get_device()
        logger.info(f"Performing high-quality OCR ({OCR_MODEL_TROCR_LARGE})...")
        ocr_pipeline = pipeline(
            "image-to-text",
            model=OCR_MODEL_TROCR_LARGE,
            device=device,
        )

        page_texts: list[str] = []
        for page_index, image in enumerate(images):
            logger.info(f"  Processing page {page_index + 1}/{len(images)}")
            try:
                ocr_result = ocr_pipeline(image)
                extracted_text = (
                    ocr_result[0].get("generated_text", "") if ocr_result else ""
                )
                page_texts.append(extracted_text)
            except Exception as e:
                logger.warning(f"  OCR failed for page {page_index + 1}: {e}")
                page_texts.append("")

        # Summarize with Mistral-7B
        logger.info(f"Summarizing text with {TEXT_MODEL_MISTRAL}...")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        summarizer_pipeline = pipeline(
            "text-generation",
            model=TEXT_MODEL_MISTRAL,
            device_map="auto",
            torch_dtype=dtype,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )

        summarized_texts: list[str] = []
        for text_index, text_content in enumerate(page_texts):
            logger.info(f"  Processing text {text_index + 1}/{len(page_texts)}")

            if len(text_content.strip()) < MIN_TEXT_LENGTH:
                summarized_texts.append(text_content)
                continue

            try:
                truncated_text = truncate_text(text_content, MAX_INPUT_LENGTH_SMALL)
                prompt = (
                    f"<s>[INST] Summarize the following text concisely:\n\n"
                    f"{truncated_text}\n\n[/INST]"
                )

                summarizer_result = summarizer_pipeline(prompt)
                summarized_output = (
                    summarizer_result[0].get("generated_text", text_content)
                    if summarizer_result
                    else text_content
                )

                # Extract answer after [/INST]
                if summarized_output and "[/INST]" in summarized_output:
                    summarized_output = summarized_output.split("[/INST]")[-1].strip()

                summarized_texts.append(summarized_output)

            except torch.cuda.OutOfMemoryError:
                logger.error("  GPU out of memory, using original text")
                summarized_texts.append(text_content)
            except Exception as e:
                logger.warning(f"  Summarization failed: {e}")
                summarized_texts.append(text_content)

        # Overlay back into PDF
        logger.info("Inserting text into PDF...")
        for page_index, summarized_text in enumerate(summarized_texts):
            if page_index >= len(document):
                break

            page = document[page_index]
            text_rect = get_page_text_rect(page, PAGE_MARGIN_PX)

            try:
                page.insert_textbox(
                    text_rect,
                    summarized_text,
                    fontsize=12,
                    color=COLOR_BLACK,
                )
            except Exception as e:
                logger.warning(f"  Failed to insert text on page {page_index + 1}: {e}")

        # Save
        logger.info(f"Saving to: {output_path}")
        document.save(str(output_path))

        logger.info(f"Done! Output saved to: {output_path}")
        return output_path

    except FileNotFoundError:
        raise
    except PDFValidationError:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise PDFProcessingError(f"Quick processing failed: {e}") from e
    finally:
        cleanup_resources(document)


def main() -> int:
    """Main entry point for quick PDF processing CLI."""
    setup_logging()

    if len(sys.argv) < 2:
        print("Quick PDF Processing Tool")
        print()
        print("Usage: python quick_pdf_process.py <input.pdf>")
        print()
        print("This will create 'edited_<input>.pdf' with summarized content.")
        print()
        print("For more options, use: python pdf_processor.py --help")
        return EXIT_SUCCESS

    pdf_file = sys.argv[1]

    # Handle help flag
    if pdf_file in ("-h", "--help"):
        print("Quick PDF Processing Tool")
        print()
        print("Usage: python quick_pdf_process.py <input.pdf>")
        print()
        print("Arguments:")
        print("  input.pdf    PDF file to process")
        print()
        print("Output:")
        print("  Creates 'edited_<input>.pdf' in the same directory")
        print()
        print("For advanced options, use: python pdf_processor.py --help")
        return EXIT_SUCCESS

    try:
        validate_pdf_file(pdf_file)
    except FileNotFoundError:
        logger.error(f"Error: File '{pdf_file}' not found")
        return EXIT_FILE_NOT_FOUND
    except PDFValidationError as e:
        logger.error(f"Error: Invalid PDF - {e}")
        return EXIT_INVALID_PDF

    try:
        output_path = quick_process(pdf_file)
        print(f"Processing complete! Output: {output_path}")
        return EXIT_SUCCESS

    except (PDFValidationError, PDFProcessingError) as e:
        logger.error(str(e))
        return EXIT_PROCESSING_ERROR
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return EXIT_PROCESSING_ERROR


if __name__ == "__main__":
    sys.exit(main())
