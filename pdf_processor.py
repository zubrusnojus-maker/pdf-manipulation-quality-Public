"""
PDF Processing Script with OCR and Text Manipulation

This script provides comprehensive PDF processing capabilities including:
- PDF loading and rasterization
- OCR (Optical Character Recognition)
- Layout analysis for tables and structures
- Text manipulation (summarization/rewriting)
- Re-insertion of modified text back into PDF

Prerequisites:
    pip install torch==2.2.0 transformers==4.44.0 pymupdf pdf2image pikepdf tqdm
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdf2image
import pikepdf
import torch
from tqdm import tqdm
from transformers import pipeline

# Constants
MIN_TEXT_LENGTH = 50

# Layout element types for document structure
LAYOUT_TYPES = {
    "header": {"fontsize_multiplier": 1.2, "align": 1},  # Centered, larger
    "paragraph": {"fontsize_multiplier": 1.0, "align": 0},  # Left-aligned, normal
    "table": {"fontsize_multiplier": 0.9, "align": 0},  # Left-aligned, slightly smaller
    "footer": {"fontsize_multiplier": 0.8, "align": 1},  # Centered, smaller
}


@dataclass
class LayoutRegion:
    """Represents a detected layout region in a document."""

    region_type: str  # header, paragraph, table, footer
    text: str
    confidence: float
    bbox: Optional[tuple] = None  # (x0, y0, x1, y1) if available


class PDFProcessor:
    """Main class for processing PDFs with OCR and text manipulation."""

    def __init__(self, model_size="small", use_layout=False):
        """
        Initialize the PDF processor.

        Args:
            model_size: "small" (mistral-7b) or "large" (llama-70b) for text processing
            use_layout: Whether to use layout analysis (slower but better for tables)
        """
        self.model_size = model_size
        self.use_layout = use_layout
        self.ocr_pipe = None
        self.layout_pipe = None
        self.summariser = None

    def load_models(self):
        """Load the required ML models (quality-optimized)."""
        print("Loading high-quality OCR model (TrOCR Large)...")
        self.ocr_pipe = pipeline(
            "image-to-text",
            model="microsoft/trocr-large-printed",
            device=0 if torch.cuda.is_available() else -1,
        )

        if self.use_layout:
            print("Loading layout model (DiT for document classification)...")
            # Use DiT (Document Image Transformer) for document structure classification
            # This is more suitable for layout analysis than document-question-answering
            self.layout_pipe = pipeline(
                "image-classification",
                model="microsoft/dit-base-finetuned-rvlcdip",
                device=0 if torch.cuda.is_available() else -1,
            )

        print(f"Loading text manipulation model ({self.model_size})...")
        if self.model_size == "large":
            # Highest quality: Llama-2-13B
            self.summariser = pipeline(
                "text-generation",
                model="meta-llama/Llama-2-13b-chat-hf",
                device_map="auto",
                model_kwargs={
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
                },
                max_new_tokens=512,
                do_sample=False,
                temperature=0.7,
            )
        else:
            # Medium quality: Mistral-7B-Instruct
            self.summariser = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.2",
                device_map="auto",
                model_kwargs={
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
                },
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
            )

    def load_pdf(self, pdf_path):
        """Load PDF document."""
        print(f"Loading PDF: {pdf_path}")
        return fitz.open(pdf_path)

    def rasterize_pdf(self, pdf_path, dpi=300):
        """Convert PDF pages to images."""
        print(f"Rasterizing PDF at {dpi} DPI...")
        return pdf2image.convert_from_path(pdf_path, dpi=dpi)

    def analyze_layout(self, image):
        """
        Analyze document layout to determine document type and structure.

        Args:
            image: PIL Image of the document page

        Returns:
            LayoutRegion with detected document type and confidence
        """
        if not self.layout_pipe:
            return LayoutRegion(
                region_type="paragraph",
                text="",
                confidence=1.0,
            )

        try:
            # Classify document type using DiT
            results = self.layout_pipe(image)

            if results and len(results) > 0:
                # Get top prediction
                top_result = results[0]
                label = top_result.get("label", "unknown").lower()
                confidence = top_result.get("score", 0.0)

                # Map RVL-CDIP labels to our layout types
                # RVL-CDIP classes: letter, form, email, handwritten, advertisement,
                # scientific_report, scientific_publication, specification, file_folder,
                # news_article, budget, invoice, presentation, questionnaire, resume, memo
                layout_mapping = {
                    "letter": "paragraph",
                    "form": "table",
                    "email": "paragraph",
                    "handwritten": "paragraph",
                    "advertisement": "header",
                    "scientific_report": "paragraph",
                    "scientific_publication": "paragraph",
                    "specification": "table",
                    "file_folder": "paragraph",
                    "news_article": "paragraph",
                    "budget": "table",
                    "invoice": "table",
                    "presentation": "header",
                    "questionnaire": "table",
                    "resume": "paragraph",
                    "memo": "paragraph",
                }

                region_type = layout_mapping.get(label, "paragraph")

                return LayoutRegion(
                    region_type=region_type,
                    text=label,  # Store original label for reference
                    confidence=confidence,
                )
            else:
                return LayoutRegion(
                    region_type="paragraph",
                    text="",
                    confidence=0.0,
                )

        except Exception as e:
            print(f"Warning: Layout analysis failed: {e}")
            return LayoutRegion(
                region_type="paragraph",
                text="",
                confidence=0.0,
            )

    def extract_text_from_images(self, images):
        """
        Extract text from rasterized PDF images using OCR.

        Args:
            images: List of PIL Images (rasterized PDF pages)

        Returns:
            List of tuples: (text, layout_region) for each page
            If use_layout is False, layout_region will be None
        """
        print("Extracting text with OCR...")
        page_data = []

        for page_img in tqdm(images, desc="OCR Processing"):
            # OCR
            ocr_result = self.ocr_pipe(page_img)

            # Extract text from result
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                text = ocr_result[0].get("generated_text", "")
            else:
                text = ""

            # Layout analysis if enabled
            layout_region = None
            if self.use_layout and text:
                layout_region = self.analyze_layout(page_img)
                if layout_region.confidence > 0.5:
                    print(
                        f"  Detected document type: {layout_region.text} "
                        f"(confidence: {layout_region.confidence:.2f})"
                    )

            page_data.append((text, layout_region))

        # For backward compatibility, extract just text if layout not used
        if not self.use_layout:
            return [text for text, _ in page_data]

        return page_data

    def manipulate_text(self, page_data, operation="summarize"):
        """
        Apply text manipulation (summarization or rewriting).

        Args:
            page_data: List of text strings OR list of (text, layout_region) tuples
            operation: "summarize" or "rewrite"

        Returns:
            List of (manipulated_text, layout_region) tuples if layout enabled,
            otherwise list of text strings
        """
        print(f"Performing text {operation}...")
        manipulated_data = []

        # Handle both formats: plain text list or (text, layout) tuples
        has_layout = (
            isinstance(page_data, list)
            and len(page_data) > 0
            and isinstance(page_data[0], tuple)
        )

        for item in tqdm(page_data, desc=f"Text {operation}"):
            if has_layout:
                text_content, layout_region = item
            else:
                text_content = item
                layout_region = None

            if not text_content or len(text_content.strip()) < MIN_TEXT_LENGTH:
                # Skip empty or very short text
                if has_layout:
                    manipulated_data.append((text_content, layout_region))
                else:
                    manipulated_data.append(text_content)
                continue

            try:
                # Truncate if text is too long for the model
                max_input_length = 2048 if self.model_size == "large" else 1024
                truncated_text = (
                    text_content[:max_input_length]
                    if len(text_content) > max_input_length
                    else text_content
                )

                # Customize prompt based on layout type if available
                layout_context = ""
                if layout_region and layout_region.confidence > 0.5:
                    doc_type = layout_region.text
                    if doc_type in ("invoice", "budget", "form"):
                        layout_context = " This is a structured document with tabular data."
                    elif doc_type in ("letter", "email", "memo"):
                        layout_context = " This is correspondence/communication."
                    elif doc_type in ("scientific_report", "scientific_publication"):
                        layout_context = " This is a scientific/technical document."

                # Format prompt for instruction-following models
                if operation == "summarize":
                    prompt = (
                        f"<s>[INST] Summarize the following text concisely "
                        f"while preserving key information.{layout_context}\n\n{truncated_text}\n\n[/INST]"
                    )
                else:
                    prompt = (
                        f"<s>[INST] Rewrite the following text to improve "
                        f"clarity and readability.{layout_context}\n\n{truncated_text}\n\n[/INST]"
                    )

                model_result = self.summariser(prompt)

                if isinstance(model_result, list) and len(model_result) > 0:
                    processed_text = model_result[0].get("generated_text", "")
                    # Extract answer after [/INST] tag
                    if "[/INST]" in processed_text:
                        processed_text = processed_text.split("[/INST]")[-1].strip()
                else:
                    processed_text = text_content

                if has_layout:
                    manipulated_data.append((processed_text, layout_region))
                else:
                    manipulated_data.append(processed_text)

            except Exception as e:
                print(f"Warning: Failed to process text: {e}")
                if has_layout:
                    manipulated_data.append((text_content, layout_region))
                else:
                    manipulated_data.append(text_content)

        return manipulated_data

    def reinsert_text(self, document, manipulated_data, fontsize=12):
        """
        Re-insert modified text back into PDF pages.

        Args:
            document: PyMuPDF document object
            manipulated_data: List of text strings OR list of (text, layout_region) tuples
            fontsize: Base font size (will be adjusted based on layout type)
        """
        print("Re-inserting text into PDF...")

        # Handle both formats: plain text list or (text, layout) tuples
        has_layout = (
            isinstance(manipulated_data, list)
            and len(manipulated_data) > 0
            and isinstance(manipulated_data[0], tuple)
        )

        for page_index, item in enumerate(tqdm(manipulated_data, desc="Inserting text")):
            if page_index >= len(document):
                break

            if has_layout:
                modified_text, layout_region = item
            else:
                modified_text = item
                layout_region = None

            page = document[page_index]

            # Clear existing text before inserting new text
            page.clean_contents()

            # Determine formatting based on layout analysis
            if layout_region and layout_region.confidence > 0.5:
                layout_type = layout_region.region_type
                layout_config = LAYOUT_TYPES.get(layout_type, LAYOUT_TYPES["paragraph"])

                adjusted_fontsize = fontsize * layout_config["fontsize_multiplier"]
                align = layout_config["align"]
            else:
                adjusted_fontsize = fontsize
                align = 0  # Left-aligned by default

            # Insert new text with layout-aware formatting
            page.insert_textbox(
                fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72),
                modified_text,
                fontsize=adjusted_fontsize,
                color=(0, 0, 0),
                align=align,
            )

    def process_pdf(self, input_path, output_path, operation="summarize", dpi=300):
        """
        Complete PDF processing pipeline.

        Args:
            input_path: Path to input PDF
            output_path: Path to save output PDF
            operation: "summarize" or "rewrite"
            dpi: DPI for rasterization
        """
        # Load models
        if not self.summariser:
            self.load_models()

        # Load PDF
        document = self.load_pdf(input_path)

        # Rasterize
        rasterized_images = self.rasterize_pdf(input_path, dpi=dpi)

        # Extract text
        extracted_page_texts = self.extract_text_from_images(rasterized_images)

        # Manipulate text
        manipulated_page_texts = self.manipulate_text(extracted_page_texts, operation=operation)

        # Re-insert text
        self.reinsert_text(document, manipulated_page_texts)

        # Save
        print(f"Saving processed PDF to: {output_path}")
        document.save(output_path)
        document.close()

        print("✓ Processing complete!")
        return output_path


def post_process_with_pikepdf(input_path, output_path):
    """
    Optional post-processing with pikepdf for PDF optimization.

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

    print(f"✓ Optimized PDF saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process PDFs with OCR and text manipulation")
    parser.add_argument("input", help="Input PDF file path")
    parser.add_argument("-o", "--output", help="Output PDF file path", default=None)
    parser.add_argument(
        "-m",
        "--model-size",
        choices=["small", "large"],
        default="small",
        help="Model size for text processing",
    )
    parser.add_argument(
        "--operation",
        choices=["summarize", "rewrite"],
        default="summarize",
        help="Text manipulation operation",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF rasterization")
    parser.add_argument("--use-layout", action="store_true", help="Enable layout analysis (slower)")
    parser.add_argument(
        "--optimize", action="store_true", help="Post-process with pikepdf optimization"
    )

    args = parser.parse_args()

    # Generate output path if not provided
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"edited_{input_path.name}")

    # Process PDF
    processor = PDFProcessor(model_size=args.model_size, use_layout=args.use_layout)

    output_path = processor.process_pdf(
        args.input, args.output, operation=args.operation, dpi=args.dpi
    )

    # Optional optimization
    if args.optimize:
        optimized_path = str(Path(output_path).parent / f"optimized_{Path(output_path).name}")
        post_process_with_pikepdf(output_path, optimized_path)


if __name__ == "__main__":
    main()
