#!/usr/bin/env python3
"""
Quick-start one-liner script for PDF processing.

Usage:
    python quick_pdf_process.py mydoc.pdf

This will create 'edited_mydoc.pdf' with summarized/rewritten content.
"""

import sys
import fitz
import pdf2image
import torch
from transformers import pipeline
from pathlib import Path


def quick_process(pdf_path):
    """Quick PDF processing in one script."""
    print(f"Processing: {pdf_path}")
    
    # 1. Load PDF & rasterise
    print("Loading and rasterizing PDF...")
    document = fitz.open(pdf_path)
    images = pdf2image.convert_from_path(pdf_path, dpi=300)
    
    # 2. OCR (High-quality)
    print("Performing high-quality OCR (TrOCR Large)...")
    ocr_pipeline = pipeline(
        "image-to-text", 
        model="microsoft/trocr-large-printed",
        device=0 if torch.cuda.is_available() else -1
    )
    page_texts = []
    for page_index, image in enumerate(images):
        print(f"  Processing page {page_index + 1}/{len(images)}")
        ocr_result = ocr_pipeline(image)
        extracted_text = ocr_result[0].get("generated_text", "") if ocr_result else ""
        page_texts.append(extracted_text)
    
    # 3. Summarise with quality model (Mistral-7B-Instruct)
    print("Summarizing text with Mistral-7B-Instruct...")
    summarizer_pipeline = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )
    
    summarized_texts = []
    for text_index, text_content in enumerate(page_texts):
        print(f"  Processing text {text_index + 1}/{len(page_texts)}")
        if len(text_content.strip()) < 50:
            summarized_texts.append(text_content)
            continue
        
        try:
            # Truncate long text
            truncated_text = text_content[:1024] if len(text_content) > 1024 else text_content
            
            # Format for instruction model
            prompt = f"<s>[INST] Summarize the following text concisely:\n\n{truncated_text}\n\n[/INST]"
            summarizer_result = summarizer_pipeline(prompt)
            
            summarized_output = summarizer_result[0].get("generated_text", text_content) if summarizer_result else text_content
            # Extract answer after [/INST]
            if "[/INST]" in summarized_output:
                summarized_output = summarized_output.split("[/INST]")[-1].strip()
            
            summarized_texts.append(summarized_output)
        except Exception as e:
            print(f"  Warning: {e}")
            summarized_texts.append(text_content)
    
    # 4. Overlay back into PDF
    print("Inserting text into PDF...")
    for page_index, summarized_text in enumerate(summarized_texts):
        if page_index >= len(document):
            break
        page = document[page_index]
        page.insert_textbox(
            fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72),
            summarized_text,
            fontsize=12,
            color=(0, 0, 0)
        )
    
    # 5. Save
    output_path = str(Path(pdf_path).parent / f"edited_{Path(pdf_path).name}")
    print(f"Saving to: {output_path}")
    document.save(output_path)
    document.close()
    
    print(f"✓ Done! Output saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_pdf_process.py <pdf_file>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    if not Path(pdf_file).exists():
        print(f"Error: File '{pdf_file}' not found")
        sys.exit(1)
    
    quick_process(pdf_file)
