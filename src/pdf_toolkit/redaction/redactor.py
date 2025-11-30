"""PDF data anonymization and redaction."""

import json
import re
from collections import defaultdict
from typing import Any

import fitz


class PDFRedactor:
    """Anonymize PDF data while preserving exact formatting."""

    def __init__(self):
        """Initialize redactor with mapping counters."""
        self.mappings: dict[str, dict[str, str]] = {}
        self.counters: defaultdict[str, int] = defaultdict(int)

    def detect_sensitive_data(self, text: str) -> list[tuple[str, str, str]]:
        """
        Detect sensitive data in text using pattern matching.

        Returns:
            List of (original_text, data_type, placeholder) tuples
        """
        detections: list[tuple[str, str, str]] = []

        # Pattern 1: Monetary amounts (e.g., 9,220.51, 12,020.51)
        money_pattern = r"\b\d{1,3}(?:,\d{3})*\.\d{2}\b"
        for match in re.finditer(money_pattern, text):
            original = match.group()
            if original not in self.mappings:
                self.counters["amount"] += 1
                placeholder = f"Amount_{chr(64 + self.counters['amount'])}"
                self.mappings[original] = {"type": "monetary_amount", "placeholder": placeholder}
            detections.append((original, "monetary_amount", self.mappings[original]["placeholder"]))

        # Pattern 2: Account numbers (8+ digits)
        account_pattern = r"\b\d{8,}\b"
        for match in re.finditer(account_pattern, text):
            original = match.group()
            if any(original in m[0] for m in detections):
                continue
            if original not in self.mappings:
                self.counters["account"] += 1
                placeholder = f"Account_{self.counters['account']}"
                self.mappings[original] = {"type": "account_number", "placeholder": placeholder}
            detections.append((original, "account_number", self.mappings[original]["placeholder"]))

        # Pattern 3: Dates (DD/MM/YYYY, DD-MM-YYYY, etc.)
        date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        for match in re.finditer(date_pattern, text):
            original = match.group()
            if original not in self.mappings:
                self.counters["date"] += 1
                placeholder = f"Date_{self.counters['date']}"
                self.mappings[original] = {"type": "date", "placeholder": placeholder}
            detections.append((original, "date", self.mappings[original]["placeholder"]))

        # Pattern 4: Company names (all caps words, 2+ words)
        company_pattern = r"\b[A-Z][A-Z\s&]{5,}(?:LTD|LIMITED|INC|CORP|GROUP)\b"
        for match in re.finditer(company_pattern, text):
            original = match.group()
            if original not in self.mappings:
                self.counters["company"] += 1
                placeholder = f"Company_{self.counters['company']}"
                self.mappings[original] = {"type": "company_name", "placeholder": placeholder}
            detections.append((original, "company_name", self.mappings[original]["placeholder"]))

        # Pattern 5: Personal names (Title + Name pattern)
        name_pattern = r"\b(?:Mr|Mrs|Ms|Dr|Miss)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
        for match in re.finditer(name_pattern, text):
            original = match.group()
            if original not in self.mappings:
                self.counters["person"] += 1
                placeholder = f"Person_{self.counters['person']}"
                self.mappings[original] = {"type": "personal_name", "placeholder": placeholder}
            detections.append((original, "personal_name", self.mappings[original]["placeholder"]))

        # Pattern 6: UK Postcodes (e.g., HA9 6BA)
        postcode_pattern = r"\b[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}\b"
        for match in re.finditer(postcode_pattern, text):
            original = match.group()
            if original not in self.mappings:
                self.counters["postcode"] += 1
                placeholder = f"Postcode_{self.counters['postcode']}"
                self.mappings[original] = {"type": "postcode", "placeholder": placeholder}
            detections.append((original, "postcode", self.mappings[original]["placeholder"]))

        # Pattern 7: Street addresses (numbers + street names)
        address_pattern = r"\b\d+\s+[A-Z][A-Za-z\s]+(?:STREET|ROAD|AVENUE|WAY|LANE|DRIVE|CLOSE)\b"
        for match in re.finditer(address_pattern, text):
            original = match.group()
            if original not in self.mappings:
                self.counters["address"] += 1
                placeholder = f"Address_{self.counters['address']}"
                self.mappings[original] = {"type": "street_address", "placeholder": placeholder}
            detections.append((original, "street_address", self.mappings[original]["placeholder"]))

        return detections

    def redact_page(self, page: fitz.Page) -> dict[str, Any]:
        """
        Redact sensitive data on a single page while preserving layout.

        Args:
            page: PyMuPDF page object

        Returns:
            Dictionary with redaction statistics
        """
        stats: dict[str, Any] = {"replacements": 0, "types": defaultdict(int)}

        text_dict = page.get_text("dict")

        for block in text_dict["blocks"]:
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    original_text: str = span["text"]
                    detections = self.detect_sensitive_data(original_text)

                    if not detections:
                        continue

                    modified_text = original_text
                    for original, data_type, placeholder in detections:
                        modified_text = modified_text.replace(original, placeholder)
                        stats["replacements"] += 1
                        stats["types"][data_type] += 1

                    bbox: Any = span["bbox"]
                    font_size: float = span["size"]
                    color: int = span.get("color", 0)

                    page.draw_rect(bbox, color=(1, 1, 1), fill=(1, 1, 1))

                    insert_point = fitz.Point(bbox[0], bbox[3])

                    try:
                        page.insert_text(
                            insert_point,
                            modified_text,
                            fontsize=font_size,
                            color=self._int_to_rgb(color),
                        )
                    except Exception as e:
                        print(f"Warning: Failed to insert text '{modified_text}': {e}")
                        page.insert_text(
                            insert_point, modified_text, fontsize=font_size, color=(0, 0, 0)
                        )

        return stats

    def _int_to_rgb(self, color_int: int) -> tuple[float, float, float]:
        """Convert integer color to RGB tuple (0-1 range)."""
        if color_int == 0:
            return (0, 0, 0)
        r = ((color_int >> 16) & 0xFF) / 255.0
        g = ((color_int >> 8) & 0xFF) / 255.0
        b = (color_int & 0xFF) / 255.0
        return (r, g, b)

    def redact_pdf(self, input_path: str, output_path: str) -> dict[str, Any]:
        """
        Redact entire PDF document.

        Args:
            input_path: Path to input PDF
            output_path: Path to save redacted PDF

        Returns:
            Dictionary with overall statistics
        """
        print(f"Loading PDF: {input_path}")
        doc: Any = fitz.open(input_path)

        overall_stats: dict[str, Any] = {
            "pages": len(doc),
            "total_replacements": 0,
            "types": defaultdict(int),
        }

        print(f"Processing {len(doc)} page(s)...")
        for page_num in range(len(doc)):
            print(f"  Redacting page {page_num + 1}/{len(doc)}")
            page = doc[page_num]
            page_stats = self.redact_page(page)

            overall_stats["total_replacements"] += page_stats["replacements"]
            for data_type, count in page_stats["types"].items():
                overall_stats["types"][data_type] += count

        print(f"Saving redacted PDF to: {output_path}")
        doc.save(output_path)
        doc.close()

        print("\nRedaction complete!")
        print(f"  Total replacements: {overall_stats['total_replacements']}")
        print("  Breakdown by type:")
        for data_type, count in overall_stats["types"].items():
            print(f"    - {data_type}: {count}")

        return overall_stats

    def save_mappings(self, output_path: str) -> None:
        """Save mapping of original values to placeholders."""
        print(f"Saving mappings to: {output_path}")
        with open(output_path, "w") as f:
            json.dump(self.mappings, f, indent=2)
