---
title: PDF Toolkit
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
tags:
  - pdf
  - redaction
  - pii
  - privacy
  - document-processing
  - mcp
  - mcp-server
short_description: PDF processing toolkit with PII detection, redaction & MCP integration
---

# 📄 PDF Toolkit

A comprehensive PDF processing toolkit with built-in MCP server support.

## Features

- 🔍 **PII Detection** - Detect sensitive information in text
- 🔒 **Text Redaction** - Replace PII with placeholders
- 📝 **PDF Text Extraction** - Extract text from PDF files
- 🔒 **PDF Redaction** - Redact sensitive data from PDFs

## MCP Integration

This Space functions as an **MCP (Model Context Protocol) server**.

### Connect from VS Code / Claude Desktop

Add to your MCP config:

```json
{
  "mcpServers": {
    "pdf-toolkit": {
      "url": "https://bischoff555-pdf-toolkit.hf.space/gradio_api/mcp/sse"
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `detect_pii` | Detect sensitive data in text |
| `redact_text` | Redact PII from text |
| `extract_text_from_pdf` | Extract text from PDF files |
| `redact_pdf` | Redact PII from PDF files |

## API

View the full API schema at: `/gradio_api/mcp/schema`

## Local Development

```bash
pip install gradio pymupdf
python app.py
```

## License

MIT
