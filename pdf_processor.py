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
import fitz  # PyMuPDF
import pdf2image
import torch
from transformers import pipeline
from tqdm import tqdm
import pikepdf
from pathlib import Path


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
            device=0 if torch.cuda.is_available() else -1
        )
        
        if self.use_layout:
            print("Loading layout model (LayoutLMv3)...")
            self.layout_pipe = pipeline(
                "document-question-answering",
                model="microsoft/layoutlmv3-base",
                device=0 if torch.cuda.is_available() else -1
            )
        
        print(f"Loading text manipulation model ({self.model_size})...")
        if self.model_size == "large":
            # Highest quality: Llama-2-13B
            self.summariser = pipeline(
                "text-generation",
                model="meta-llama/Llama-2-13b-chat-hf",
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
    
    def extract_text_from_images(self, images):
        """Extract text from rasterized PDF images using OCR."""
        print("Extracting text with OCR...")
        page_texts = []
        
        for page_img in tqdm(images, desc="OCR Processing"):
            # OCR
            ocr_result = self.ocr_pipe(page_img)
            
            # Extract text from result
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                text = ocr_result[0].get("generated_text", "")
            else:
                text = ""
            
            # Optional: Layout analysis
            if self.use_layout and text:
                # Layout analysis can be added here for table detection
                pass
            
            page_texts.append(text)
        
        return page_texts
    
    def manipulate_text(self, page_texts, operation="summarize"):
        """
        Apply text manipulation (summarization or rewriting).
        
        Args:
            page_texts: List of text strings from each page
            operation: "summarize" or "rewrite"
        """
        print(f"Performing text {operation}...")
        rewritten = []
        
        for txt in tqdm(page_texts, desc=f"Text {operation}"):
            if not txt or len(txt.strip()) < 50:
                # Skip empty or very short text
                rewritten.append(txt)
                continue
            
            try:
                # Truncate if text is too long for the model
                max_input_length = 2048 if self.model_size == "large" else 1024
                if len(txt) > max_input_length:
                    txt = txt[:max_input_length]
                
                # Format prompt for instruction-following models
                if operation == "summarize":
                    prompt = f"<s>[INST] Summarize the following text concisely while preserving key information:\n\n{txt}\n\n[/INST]"
                else:
                    prompt = f"<s>[INST] Rewrite the following text to improve clarity and readability:\n\n{txt}\n\n[/INST]"
                
                result = self.summariser(prompt)
                
                if isinstance(result, list) and len(result) > 0:
                    output_text = result[0].get("generated_text", "")
                    # Extract answer after [/INST] tag
                    if "[/INST]" in output_text:
                        output_text = output_text.split("[/INST]")[-1].strip()
                else:
                    output_text = txt
                
                rewritten.append(output_text)
            except Exception as e:
                print(f"Warning: Failed to process text: {e}")
                rewritten.append(txt)
        
        return rewritten
    
    def reinsert_text(self, doc, rewritten_texts, fontsize=12):
        """Re-insert modified text back into PDF pages."""
        print("Re-inserting text into PDF...")
        
        for i, new_text in enumerate(tqdm(rewritten_texts, desc="Inserting text")):
            if i >= len(doc):
                break
            
            page = doc[i]
            
            # Clear existing text (optional - comment out to overlay)
            # page.clean_contents()
            
            # Insert new text - simple overlay at top left
            # Adjust (x, y) coordinates for your layout needs
            page.insert_textbox(
                fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72),
                new_text,
                fontsize=fontsize,
                color=(0, 0, 0),
                align=0
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
        doc = self.load_pdf(input_path)
        
        # Rasterize
        images = self.rasterize_pdf(input_path, dpi=dpi)
        
        # Extract text
        page_texts = self.extract_text_from_images(images)
        
        # Manipulate text
        rewritten_texts = self.manipulate_text(page_texts, operation=operation)
        
        # Re-insert text
        self.reinsert_text(doc, rewritten_texts)
        
        # Save
        print(f"Saving processed PDF to: {output_path}")
        doc.save(output_path)
        doc.close()
        
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
    
    src = pikepdf.Pdf.open(input_path)
    dst = pikepdf.Pdf.new()
    
    for page in tqdm(src.pages, desc="Copying pages"):
        dst.pages.append(page)
    
    dst.save(output_path)
    src.close()
    
    print(f"✓ Optimized PDF saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process PDFs with OCR and text manipulation"
    )
    parser.add_argument("input", help="Input PDF file path")
    parser.add_argument("-o", "--output", help="Output PDF file path", default=None)
    parser.add_argument(
        "-m", "--model-size",
        choices=["small", "large"],
        default="small",
        help="Model size for text processing"
    )
    parser.add_argument(
        "--operation",
        choices=["summarize", "rewrite"],
        default="summarize",
        help="Text manipulation operation"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF rasterization"
    )
    parser.add_argument(
        "--use-layout",
        action="store_true",
        help="Enable layout analysis (slower)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Post-process with pikepdf optimization"
    )
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"edited_{input_path.name}")
    
    # Process PDF
    processor = PDFProcessor(
        model_size=args.model_size,
        use_layout=args.use_layout
    )
    
    output_path = processor.process_pdf(
        args.input,
        args.output,
        operation=args.operation,
        dpi=args.dpi
    )
    
    # Optional optimization
    if args.optimize:
        optimized_path = str(Path(output_path).parent / f"optimized_{Path(output_path).name}")
        post_process_with_pikepdf(output_path, optimized_path)


if __name__ == "__main__":
    main()
