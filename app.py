"""PDF Toolkit Hugging Face Space with Gradio + MCP support."""

import tempfile
from collections import defaultdict

import gradio as gr

from pdf_toolkit.redaction.patterns import REDACTION_PATTERNS
from pdf_toolkit.redaction.redactor import PDFRedactor


def _format_detection_results(groups: dict[str, list[tuple[str, str]]]) -> str:
    """Format grouped detection results as Markdown."""

    total = sum(len(items) for items in groups.values())
    result = "## PII Detection Results\n\n"
    result += f"**Total detections:** {total}\n\n"

    if total == 0:
        result += "_No sensitive data detected._\n"
        return result

    for data_type in REDACTION_PATTERNS.keys():
        items = groups.get(data_type, [])
        if not items:
            continue

        heading = data_type.replace("_", " ").title()
        result += f"### {heading} ({len(items)})\n"
        for original, placeholder in items:
            result += f"- `{original}` → `{placeholder}`\n"
        result += "\n"

    return result


def detect_pii(text: str) -> str:
    """Detect and display sensitive data in free-form text."""

    if not text or not text.strip():
        return "_Enter text to analyze for PII._"

    redactor = PDFRedactor()
    detections = redactor.detect_sensitive_data(text)

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for original, data_type, placeholder in detections:
        grouped[data_type].append((original, placeholder))

    return _format_detection_results(grouped)


def redact_text(text: str, preserve_dates: bool = False) -> str:
    """Redact sensitive information from text using PDFRedactor logic."""

    if not text:
        return ""

    redactor = PDFRedactor()
    detections = redactor.detect_sensitive_data(text, preserve_dates=preserve_dates)

    redacted_text = text
    for original, _, placeholder in detections:
        redacted_text = redacted_text.replace(original, placeholder)

    return redacted_text


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from a PDF file.

    Uses PyMuPDF for text extraction. For scanned PDFs,
    consider using the OCR endpoint.

    Args:
        pdf_file: Uploaded PDF file

    Returns:
        Extracted text from all pages
    """
    import fitz

    if pdf_file is None:
        return "Please upload a PDF file."

    try:
        doc = fitz.open(pdf_file.name)
        text_parts = []

        for page_num, page in enumerate(doc):  # type: ignore
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

        doc.close()

        if text_parts:
            return "\n\n".join(text_parts)
        else:
            return "No text found in PDF. This may be a scanned document - try OCR extraction."

    except Exception as e:
        return f"Error extracting text: {str(e)}"


def redact_pdf(pdf_file, preserve_dates: bool = False):
    """
    Redact sensitive information from a PDF file.

    Detects and replaces PII with placeholders, returns redacted PDF.

    Args:
        pdf_file: Uploaded PDF file
        preserve_dates: If True, don't redact dates

    Returns:
        Tuple of (redacted PDF path, summary text)
    """
    if pdf_file is None:
        return None, "Please upload a PDF file."

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_output:
            output_path = tmp_output.name

        redactor = PDFRedactor()
        stats = redactor.redact_pdf(pdf_file.name, output_path, preserve_dates=preserve_dates)

        summary = "## Redaction Complete\n\n"
        summary += f"- **Pages processed:** {stats['pages']}\n"
        summary += f"- **Items redacted:** {stats['total_replacements']}\n"

        type_counts = {k: v for k, v in stats["types"].items() if v}
        if type_counts:
            summary += "\n### Breakdown by type\n"
            for data_type in REDACTION_PATTERNS.keys():
                count = type_counts.get(data_type)
                if not count:
                    continue
                summary += f"- {data_type.replace('_', ' ').title()}: {count}\n"

        return output_path, summary

    except Exception as e:
        return None, f"Error: {str(e)}"


# Create Gradio Interface
with gr.Blocks(title="PDF Toolkit") as demo:
    gr.Markdown("""
    # 📄 PDF Toolkit
    
    A comprehensive toolkit for PDF processing with PII detection and redaction.
    
    **Features:**
    - 🔍 Detect sensitive data (PII) in text
    - 🔒 Redact PII from text and PDFs
    - 📝 Extract text from PDFs
    
    ---
    """)

    with gr.Tabs():
        # Tab 1: PII Detection
        with gr.TabItem("🔍 Detect PII"):
            gr.Markdown("Analyze text for sensitive information without modifying it.")
            with gr.Row():
                with gr.Column():
                    detect_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter text to analyze for PII...",
                        lines=10,
                    )
                    detect_btn = gr.Button("Detect PII", variant="primary")
                with gr.Column():
                    detect_output = gr.Markdown(label="Detection Results")

            detect_btn.click(detect_pii, inputs=detect_input, outputs=detect_output)

            gr.Examples(
                examples=[
                    [
                        "Invoice Date: 30/09/2021\nAccount: 12345678\nAmount: £9,220.51\nCompany: ACME TESTING LIMITED\nContact: Mr John Smith\nAddress: 123 High Street, HA9 6BA"
                    ],
                    [
                        "Payment of 15,000.00 received from Mrs Jane Doe on 01/12/2024 for account 98765432."
                    ],
                ],
                inputs=detect_input,
            )

        # Tab 2: Text Redaction
        with gr.TabItem("🔒 Redact Text"):
            gr.Markdown("Replace sensitive information with placeholders.")
            with gr.Row():
                with gr.Column():
                    redact_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter text to redact...",
                        lines=10,
                    )
                    preserve_dates_text = gr.Checkbox(label="Preserve dates", value=False)
                    redact_btn = gr.Button("Redact Text", variant="primary")
                with gr.Column():
                    redact_output = gr.Textbox(label="Redacted Text", lines=10)

            redact_btn.click(
                redact_text, inputs=[redact_input, preserve_dates_text], outputs=redact_output
            )

        # Tab 3: PDF Text Extraction
        with gr.TabItem("📝 Extract PDF Text"):
            gr.Markdown("Extract text content from PDF files.")
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                    extract_btn = gr.Button("Extract Text", variant="primary")
                with gr.Column():
                    extract_output = gr.Textbox(label="Extracted Text", lines=15)

            extract_btn.click(extract_text_from_pdf, inputs=pdf_input, outputs=extract_output)

        # Tab 4: PDF Redaction
        with gr.TabItem("🔒 Redact PDF"):
            gr.Markdown("Redact sensitive information from PDF files.")
            with gr.Row():
                with gr.Column():
                    pdf_redact_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                    preserve_dates_pdf = gr.Checkbox(label="Preserve dates", value=False)
                    pdf_redact_btn = gr.Button("Redact PDF", variant="primary")
                with gr.Column():
                    pdf_redact_output = gr.File(label="Download Redacted PDF")
                    pdf_redact_summary = gr.Markdown(label="Summary")

            pdf_redact_btn.click(
                redact_pdf,
                inputs=[pdf_redact_input, preserve_dates_pdf],
                outputs=[pdf_redact_output, pdf_redact_summary],
            )

    gr.Markdown("""
    ---
    
    ### 🔌 MCP Integration
    
    This Space is available as an MCP server! Add to your MCP client:
    
    ```json
    {
      "mcpServers": {
        "pdf-toolkit": {
          "url": "https://YOUR-SPACE-URL/gradio_api/mcp/sse"
        }
      }
    }
    ```
    
    [View API Documentation](/gradio_api/mcp/schema)
    """)

# Launch with MCP server enabled
if __name__ == "__main__":
    demo.launch(mcp_server=True)
