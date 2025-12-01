"""Model loading utilities for OCR, layout analysis, and text generation."""

import torch
from transformers import pipeline

from pdf_toolkit.core.constants import get_model_with_revision


class ModelLoader:
    """Handles loading and configuration of ML models."""

    def __init__(self):
        """Initialize model loader."""
        self.device = 0 if torch.cuda.is_available() else -1
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def load_ocr_model(self):
        """Load the TrOCR model for optical character recognition with pinned version."""
        print("Loading high-quality OCR model (TrOCR Large)...")
        from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline

        model = AutoModelForImageTextToText.from_pretrained("microsoft/trocr-large-printed")
        processor = AutoProcessor.from_pretrained("microsoft/trocr-large-printed")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-large-printed")
        # Some transformers versions expect an image processor instead of a feature extractor
        try:
            from transformers import AutoImageProcessor

            image_processor = AutoImageProcessor.from_pretrained("microsoft/trocr-large-printed")
        except Exception:
            image_processor = getattr(processor, "feature_extractor", None)

        return pipeline(
            "image-to-text",
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=self.device,
        )

    def load_layout_model(self):
        """Load the DiT model for document layout classification with pinned version."""
        print("Loading layout model (DiT for document classification)...")
        try:
            return pipeline(
                "image-classification",
                **get_model_with_revision("layout"),
                device=self.device,
            )
        except Exception as e:
            print(f"Warning: could not load layout model: {e}\nProceeding without layout model.")
            return None

    def load_text_model(self, model_size="small"):
        """
        Load the text generation model for summarization/rewriting.

        Args:
            model_size: "small" for Mistral-7B, "large" for Llama-2-13B
        """
        print(f"Loading text manipulation model ({model_size})...")

        if model_size == "large":
            return pipeline(
                "text-generation",
                **get_model_with_revision("text_large"),
                device_map="auto",
                model_kwargs={"dtype": self.torch_dtype},
                max_new_tokens=512,
                do_sample=False,
                temperature=0.7,
            )
        else:
            return pipeline(
                "text-generation",
                **get_model_with_revision("text_small"),
                device_map="auto",
                model_kwargs={"dtype": self.torch_dtype},
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
            )
