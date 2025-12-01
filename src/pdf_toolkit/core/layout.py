"""Layout analysis configuration and data structures."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LayoutRegion:
    """Represents a detected layout region in a document."""

    region_type: str  # header, paragraph, table, footer
    text: str
    confidence: float
    bbox: Optional[tuple] = None  # (x0, y0, x1, y1) if available


# Layout element types for document structure formatting
LAYOUT_TYPES = {
    "header": {"fontsize_multiplier": 1.2, "align": 1},  # Centered, larger
    "paragraph": {"fontsize_multiplier": 1.0, "align": 0},  # Left-aligned, normal
    "table": {"fontsize_multiplier": 0.9, "align": 0},  # Left-aligned, slightly smaller
    "footer": {"fontsize_multiplier": 0.8, "align": 1},  # Centered, smaller
}

# Mapping from RVL-CDIP document classes to layout types
DOCUMENT_TYPE_MAPPING = {
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
