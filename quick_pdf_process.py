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
    doc = fitz.open(pdf_path)
    imgs = pdf2image.convert_from_path(pdf_path, dpi=300)
    
    # 2. OCR (High-quality)
    print("Performing high-quality OCR (TrOCR Large)...")
    ocr = pipeline(
        "image-to-text", 
        model="microsoft/trocr-large-printed",
        device=0 if torch.cuda.is_available() else -1
    )
    texts = []
    for i, img in enumerate(imgs):
        print(f"  Processing page {i+1}/{len(imgs)}")
        result = ocr(img)
        text = result[0].get("generated_text", "") if result else ""
        texts.append(text)
    
    # 3. Summarise with quality model (Mistral-7B-Instruct)
    print("Summarizing text with Mistral-7B-Instruct...")
    summ = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )
    
    rewrites = []
    for i, t in enumerate(texts):
        print(f"  Processing text {i+1}/{len(texts)}")
        if len(t.strip()) < 50:
            rewrites.append(t)
            continue
        
        try:
            # Truncate long text
            if len(t) > 1024:
                t = t[:1024]
            
            # Format for instruction model
            prompt = f"<s>[INST] Summarize the following text concisely:\n\n{t}\n\n[/INST]"
            result = summ(prompt)
            
            output = result[0].get("generated_text", t) if result else t
            # Extract answer after [/INST]
            if "[/INST]" in output:
                output = output.split("[/INST]")[-1].strip()
            
            rewrites.append(output)
        except Exception as e:
            print(f"  Warning: {e}")
            rewrites.append(t)
    
    # 4. Overlay back into PDF
    print("Inserting text into PDF...")
    for i, new_t in enumerate(rewrites):
        if i >= len(doc):
            break
        page = doc[i]
        page.insert_textbox(
            fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72),
            new_t,
            fontsize=12,
            color=(0, 0, 0)
        )
    
    # 5. Save
    output_path = str(Path(pdf_path).parent / f"edited_{Path(pdf_path).name}")
    print(f"Saving to: {output_path}")
    doc.save(output_path)
    doc.close()
    
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
