"""Main PDF processor class for OCR and text manipulation."""

import fitz
from tqdm import tqdm

from pdf_toolkit.core.constants import (
    MIN_TEXT_LENGTH,
    MAX_INPUT_LENGTH_SMALL,
    MAX_INPUT_LENGTH_LARGE,
    MARGIN_POINTS,
)
from pdf_toolkit.core.layout import LayoutRegion, LAYOUT_TYPES, DOCUMENT_TYPE_MAPPING
from pdf_toolkit.models.loader import ModelLoader
from pdf_toolkit.utils.pdf_utils import rasterize_pdf


class PDFProcessor:
    """Main class for processing PDFs with OCR and text manipulation."""

    def __init__(self, model_size="small", use_layout=False):
        """
        Initialize the PDF processor.

        Args:
            model_size: "small" (Mistral-7B) or "large" (Llama-2-13B) for text processing
            use_layout: Whether to use layout analysis for document classification
        """
        self.model_size = model_size
        self.use_layout = use_layout
        self.model_loader = ModelLoader()
        self.ocr_pipe = None
        self.layout_pipe = None
        self.summariser = None

    def load_models(self):
        """Load the required ML models."""
        self.ocr_pipe = self.model_loader.load_ocr_model()

        if self.use_layout:
            self.layout_pipe = self.model_loader.load_layout_model()

        self.summariser = self.model_loader.load_text_model(self.model_size)

    def load_pdf(self, pdf_path):
        """Load PDF document."""
        print(f"Loading PDF: {pdf_path}")
        return fitz.open(pdf_path)

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
            results = self.layout_pipe(image)

            if results and len(results) > 0:
                top_result = results[0]
                label = top_result.get("label", "unknown").lower()
                confidence = top_result.get("score", 0.0)

                region_type = DOCUMENT_TYPE_MAPPING.get(label, "paragraph")

                return LayoutRegion(
                    region_type=region_type,
                    text=label,
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
            ocr_result = self.ocr_pipe(page_img)

            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                text = ocr_result[0].get("generated_text", "")
            else:
                text = ""

            layout_region = None
            if self.use_layout and text:
                layout_region = self.analyze_layout(page_img)
                if layout_region.confidence > 0.5:
                    print(
                        f"  Detected document type: {layout_region.text} "
                        f"(confidence: {layout_region.confidence:.2f})"
                    )

            page_data.append((text, layout_region))

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

        has_layout = (
            isinstance(page_data, list)
            and len(page_data) > 0
            and isinstance(page_data[0], tuple)
        )

        max_input_length = (
            MAX_INPUT_LENGTH_LARGE if self.model_size == "large" else MAX_INPUT_LENGTH_SMALL
        )

        for item in tqdm(page_data, desc=f"Text {operation}"):
            if has_layout:
                text_content, layout_region = item
            else:
                text_content = item
                layout_region = None

            if not text_content or len(text_content.strip()) < MIN_TEXT_LENGTH:
                if has_layout:
                    manipulated_data.append((text_content, layout_region))
                else:
                    manipulated_data.append(text_content)
                continue

            try:
                truncated_text = (
                    text_content[:max_input_length]
                    if len(text_content) > max_input_length
                    else text_content
                )

                layout_context = self._get_layout_context(layout_region)

                prompt = self._build_prompt(operation, truncated_text, layout_context)
                model_result = self.summariser(prompt)

                if isinstance(model_result, list) and len(model_result) > 0:
                    processed_text = model_result[0].get("generated_text", "")
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

    def _get_layout_context(self, layout_region):
        """Get additional context string based on layout type."""
        if not layout_region or layout_region.confidence <= 0.5:
            return ""

        doc_type = layout_region.text
        if doc_type in ("invoice", "budget", "form"):
            return " This is a structured document with tabular data."
        elif doc_type in ("letter", "email", "memo"):
            return " This is correspondence/communication."
        elif doc_type in ("scientific_report", "scientific_publication"):
            return " This is a scientific/technical document."
        return ""

    def _build_prompt(self, operation, text, layout_context):
        """Build instruction prompt for the text model."""
        if operation == "summarize":
            return (
                f"<s>[INST] Summarize the following text concisely "
                f"while preserving key information.{layout_context}\n\n{text}\n\n[/INST]"
            )
        else:
            return (
                f"<s>[INST] Rewrite the following text to improve "
                f"clarity and readability.{layout_context}\n\n{text}\n\n[/INST]"
            )

    def reinsert_text(self, document, manipulated_data, fontsize=12):
        """
        Re-insert modified text back into PDF pages.

        Args:
            document: PyMuPDF document object
            manipulated_data: List of text strings OR list of (text, layout_region) tuples
            fontsize: Base font size (will be adjusted based on layout type)
        """
        print("Re-inserting text into PDF...")

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
            page.clean_contents()

            if layout_region and layout_region.confidence > 0.5:
                layout_type = layout_region.region_type
                layout_config = LAYOUT_TYPES.get(layout_type, LAYOUT_TYPES["paragraph"])
                adjusted_fontsize = fontsize * layout_config["fontsize_multiplier"]
                align = layout_config["align"]
            else:
                adjusted_fontsize = fontsize
                align = 0

            page.insert_textbox(
                fitz.Rect(
                    MARGIN_POINTS,
                    MARGIN_POINTS,
                    page.rect.width - MARGIN_POINTS,
                    page.rect.height - MARGIN_POINTS,
                ),
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
        if not self.summariser:
            self.load_models()

        document = self.load_pdf(input_path)
        rasterized_images = rasterize_pdf(input_path, dpi=dpi)
        extracted_page_texts = self.extract_text_from_images(rasterized_images)
        manipulated_page_texts = self.manipulate_text(extracted_page_texts, operation=operation)
        self.reinsert_text(document, manipulated_page_texts)

        print(f"Saving processed PDF to: {output_path}")
        document.save(output_path)
        document.close()

        print("Processing complete!")
        return output_path
