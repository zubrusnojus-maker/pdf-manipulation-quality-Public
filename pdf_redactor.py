#!/usr/bin/env python3
"""
PDF Data Anonymization/Redaction Tool

A production-ready tool for replacing sensitive data (numbers, names, emails,
phone numbers, etc.) with placeholder names while preserving exact document
formatting, layout, lines, headers, and logos.

Features:
    - Comprehensive pattern matching for 20+ sensitive data types
    - Input validation and secure file handling
    - Metadata redaction for complete privacy
    - Secure mapping file storage with warnings
    - Post-redaction verification

Usage:
    python pdf_redactor.py input.pdf -o output.pdf
    python pdf_redactor.py input.pdf --mapping-output mappings.json
    python pdf_redactor.py input.pdf --skip-patterns email,phone_uk
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import fitz  # PyMuPDF # type: ignore[import-untyped]

from constants import (
    COLOR_BLACK,
    COLOR_WHITE,
    EXIT_FILE_NOT_FOUND,
    EXIT_INVALID_PDF,
    EXIT_PROCESSING_ERROR,
    EXIT_SUCCESS,
    MAPPING_FILE_PERMISSIONS,
    MAPPING_FILE_WARNING,
    PATTERN_EXCLUSIONS,
    REDACTION_PATTERNS,
)
from pdf_utils import (
    PDFProcessingError,
    PDFValidationError,
    cleanup_resources,
    int_to_rgb,
    set_secure_file_permissions,
    setup_logging,
    validate_output_path,
    validate_pdf_file,
)

# Configure module logger
logger = logging.getLogger(__name__)


class RedactionStats:
    """Track redaction statistics."""

    def __init__(self) -> None:
        self.pages_processed: int = 0
        self.total_replacements: int = 0
        self.types: defaultdict[str, int] = defaultdict(int)
        self.patterns_matched: set[str] = set()
        self.verification_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "pages_processed": self.pages_processed,
            "total_replacements": self.total_replacements,
            "types": dict(self.types),
            "patterns_matched": list(self.patterns_matched),
            "verification_passed": self.verification_passed,
        }


class PDFRedactor:
    """Anonymize PDF data while preserving exact formatting."""

    def __init__(
        self,
        skip_patterns: set[str] | None = None,
        custom_patterns: dict[str, dict[str, str]] | None = None,
        verify_redaction: bool = True,
    ) -> None:
        """
        Initialize redactor with configuration.

        Args:
            skip_patterns: Set of pattern names to skip
            custom_patterns: Additional custom patterns to use
            verify_redaction: Whether to verify redaction after processing
        """
        self.mappings: dict[str, dict[str, str]] = {}
        self.counters: defaultdict[str, int] = defaultdict(int)
        self.skip_patterns: set[str] = skip_patterns or set()
        self.verify_redaction = verify_redaction

        # Build active patterns
        self.patterns: dict[str, dict[str, str]] = {}
        for name, config in REDACTION_PATTERNS.items():
            if name not in self.skip_patterns:
                self.patterns[name] = config

        # Add custom patterns
        if custom_patterns:
            for name, config in custom_patterns.items():
                if "pattern" in config and "prefix" in config:
                    self.patterns[name] = config

        logger.info(f"Initialized with {len(self.patterns)} active patterns")

    def _get_placeholder(self, pattern_name: str, original: str) -> str:
        """
        Get or create a placeholder for a detected value.

        Args:
            pattern_name: Name of the pattern that matched
            original: Original text that was matched

        Returns:
            Placeholder string
        """
        if original in self.mappings:
            return self.mappings[original]["placeholder"]

        prefix = self.patterns[pattern_name]["prefix"]
        self.counters[pattern_name] += 1
        count = self.counters[pattern_name]

        # Use letters for first 26, then numbers
        if count <= 26:
            suffix = chr(64 + count)  # A, B, C, ...
        else:
            suffix = str(count)

        placeholder = f"{prefix}{suffix}"
        self.mappings[original] = {
            "type": pattern_name,
            "placeholder": placeholder,
            "detected_at": datetime.now().isoformat(),
        }

        return placeholder

    def _should_skip_match(
        self,
        pattern_name: str,
        match_text: str,
        existing_detections: list[tuple[str, str, str]],
    ) -> bool:
        """
        Check if a match should be skipped due to overlap with existing detections.

        Args:
            pattern_name: Name of current pattern
            match_text: Text that was matched
            existing_detections: List of existing (original, type, placeholder) tuples

        Returns:
            True if match should be skipped
        """
        # Check for pattern exclusions
        exclusions = PATTERN_EXCLUSIONS.get(pattern_name, [])
        for original, data_type, _ in existing_detections:
            # Skip if this text is part of an already-matched value
            if match_text in original or original in match_text:
                return True
            # Skip if excluded by pattern rules
            if data_type in exclusions:
                if match_text in original:
                    return True

        return False

    def detect_sensitive_data(self, text: str) -> list[tuple[str, str, str]]:
        """
        Detect sensitive data in text using comprehensive pattern matching.

        Args:
            text: Text to scan for sensitive data

        Returns:
            List of (original_text, data_type, placeholder) tuples
        """
        detections: list[tuple[str, str, str]] = []

        for pattern_name, config in self.patterns.items():
            pattern = config["pattern"]

            try:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    original = match.group()

                    # Skip if already detected or overlaps with existing
                    if self._should_skip_match(pattern_name, original, detections):
                        continue

                    placeholder = self._get_placeholder(pattern_name, original)
                    detections.append((original, pattern_name, placeholder))

            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern_name}': {e}")
                continue

        return detections

    def redact_page(self, page: fitz.Page) -> dict[str, Any]:  # type: ignore[name-defined]
        """
        Redact sensitive data on a single page while preserving layout.

        Args:
            page: PyMuPDF page object

        Returns:
            Dictionary with redaction statistics
        """
        stats: dict[str, Any] = {
            "replacements": 0,
            "types": defaultdict(int),
            "errors": [],
        }

        # Extract text with detailed positioning information
        try:
            text_dict = page.get_text("dict")
        except Exception as e:
            logger.error(f"Failed to extract text from page: {e}")
            stats["errors"].append(str(e))
            return stats

        # Process each text block
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # Skip non-text blocks (images, etc.)
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    original_text: str = span.get("text", "")

                    if not original_text.strip():
                        continue

                    # Detect sensitive data
                    detections = self.detect_sensitive_data(original_text)

                    if not detections:
                        continue

                    # Replace sensitive data
                    modified_text = original_text
                    for original, data_type, placeholder in detections:
                        modified_text = modified_text.replace(original, placeholder)
                        stats["replacements"] += 1
                        stats["types"][data_type] += 1

                    # Get position and font information
                    bbox = span.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue

                    font_size: float = span.get("size", 12.0)
                    color: int = span.get("color", 0)

                    # Remove original text by drawing white rectangle
                    try:
                        page.draw_rect(
                            bbox,
                            color=COLOR_WHITE,
                            fill=COLOR_WHITE,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to draw redaction rect: {e}")
                        stats["errors"].append(f"draw_rect: {e}")
                        continue

                    # Insert replacement text at original position
                    insert_point = fitz.Point(bbox[0], bbox[3])

                    try:
                        page.insert_text(
                            insert_point,
                            modified_text,
                            fontsize=font_size,
                            color=int_to_rgb(color),
                        )
                    except Exception as e:
                        logger.warning(f"Failed to insert text, using fallback: {e}")
                        try:
                            # Fallback: use default font and black color
                            page.insert_text(
                                insert_point,
                                modified_text,
                                fontsize=font_size,
                                color=COLOR_BLACK,
                            )
                        except Exception as fallback_error:
                            stats["errors"].append(f"insert_text: {fallback_error}")

        return stats

    def redact_metadata(self, doc: fitz.Document) -> None:  # type: ignore[name-defined]
        """
        Redact PDF metadata to remove potentially sensitive information.

        Args:
            doc: PyMuPDF document object
        """
        logger.info("Redacting document metadata...")

        try:
            # Clear standard metadata fields
            doc.set_metadata({
                "title": "[REDACTED]",
                "author": "[REDACTED]",
                "subject": "[REDACTED]",
                "keywords": "[REDACTED]",
                "creator": "[REDACTED]",
                "producer": "PDF Redaction Tool",
                "creationDate": "",
                "modDate": "",
            })
        except Exception as e:
            logger.warning(f"Could not fully redact metadata: {e}")

        # Remove annotations from all pages
        try:
            for page in doc:
                annots = list(page.annots()) if page.annots() else []
                for annot in annots:
                    page.delete_annot(annot)
        except Exception as e:
            logger.warning(f"Could not remove annotations: {e}")

    def verify_redaction_complete(
        self,
        doc: fitz.Document,  # type: ignore[name-defined]
    ) -> tuple[bool, list[str]]:
        """
        Verify that all sensitive data patterns have been redacted.

        Args:
            doc: PyMuPDF document to verify

        Returns:
            Tuple of (success, list of unredacted patterns found)
        """
        logger.info("Verifying redaction completeness...")
        unredacted: list[str] = []

        for page_num, page in enumerate(doc):
            try:
                text = page.get_text()

                for pattern_name, config in self.patterns.items():
                    pattern = config["pattern"]
                    matches = re.findall(pattern, text, re.IGNORECASE)

                    # Filter out placeholders (they contain the prefix)
                    prefix = config["prefix"]
                    real_matches = [m for m in matches if prefix not in m]

                    if real_matches:
                        for match in real_matches[:3]:  # Report first 3
                            msg = f"Page {page_num + 1}: Unredacted {pattern_name}: '{match}'"
                            unredacted.append(msg)
                            logger.warning(msg)

            except Exception as e:
                logger.warning(f"Verification failed for page {page_num + 1}: {e}")

        success = len(unredacted) == 0
        if success:
            logger.info("Verification passed: No unredacted sensitive data found")
        else:
            logger.warning(f"Verification found {len(unredacted)} potential issues")

        return success, unredacted

    def redact_pdf(
        self,
        input_path: str | Path,
        output_path: str | Path,
        redact_metadata: bool = True,
    ) -> RedactionStats:
        """
        Redact entire PDF document.

        Args:
            input_path: Path to input PDF
            output_path: Path to save redacted PDF
            redact_metadata: Whether to redact document metadata

        Returns:
            RedactionStats object with processing statistics

        Raises:
            PDFValidationError: If input file is invalid
            PDFProcessingError: If processing fails
        """
        stats = RedactionStats()
        doc = None

        try:
            # Validate input
            validated_input = validate_pdf_file(input_path)
            validated_output = validate_output_path(output_path, input_path)

            logger.info(f"Loading PDF: {validated_input}")
            doc = fitz.open(str(validated_input))

            stats.pages_processed = len(doc)
            logger.info(f"Processing {len(doc)} page(s)...")

            for page_num in range(len(doc)):
                logger.info(f"  Redacting page {page_num + 1}/{len(doc)}")
                page = doc[page_num]
                page_stats = self.redact_page(page)

                stats.total_replacements += page_stats["replacements"]
                for data_type, count in page_stats["types"].items():
                    stats.types[data_type] += count
                    stats.patterns_matched.add(data_type)

            # Redact metadata
            if redact_metadata:
                self.redact_metadata(doc)

            # Verify redaction if enabled
            if self.verify_redaction:
                stats.verification_passed, _ = self.verify_redaction_complete(doc)

            # Save the redacted PDF
            logger.info(f"Saving redacted PDF to: {validated_output}")
            doc.save(str(validated_output))

            logger.info("Redaction complete!")
            logger.info(f"  Total replacements: {stats.total_replacements}")
            logger.info("  Breakdown by type:")
            for data_type, count in sorted(stats.types.items()):
                logger.info(f"    - {data_type}: {count}")

            return stats

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except PDFValidationError as e:
            logger.error(f"PDF validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during redaction: {e}")
            raise PDFProcessingError(f"Redaction failed: {e}") from e
        finally:
            cleanup_resources(doc)

    def save_mappings(
        self,
        output_path: str | Path,
        include_warning: bool = True,
        set_permissions: bool = True,
    ) -> None:
        """
        Save mapping of original values to placeholders.

        Args:
            output_path: Path to save JSON mappings
            include_warning: Include security warning in file
            set_permissions: Set restrictive file permissions

        Warning:
            The mapping file contains all original sensitive data.
            Store securely and delete when no longer needed.
        """
        output_path = Path(output_path)
        logger.info(f"Saving mappings to: {output_path}")

        # Prepare output data
        output_data: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "total_mappings": len(self.mappings),
            "mappings": self.mappings,
        }

        if include_warning:
            output_data["WARNING"] = MAPPING_FILE_WARNING.strip()

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            # Set secure permissions
            if set_permissions:
                set_secure_file_permissions(output_path)

            logger.warning(
                "SECURITY WARNING: Mapping file contains original sensitive data. "
                "Store securely and delete when no longer needed."
            )

        except (IOError, OSError) as e:
            logger.error(f"Failed to save mappings: {e}")
            raise PDFProcessingError(f"Could not save mappings: {e}") from e


def main() -> int:
    """Main entry point for PDF redaction CLI."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Anonymize PDF data while preserving formatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pdf_redactor.py document.pdf
    python pdf_redactor.py document.pdf -o redacted.pdf
    python pdf_redactor.py document.pdf --skip-patterns email,phone_uk
    python pdf_redactor.py document.pdf --no-verify --no-metadata
        """,
    )

    parser.add_argument(
        "input",
        help="Input PDF file path",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output PDF file path (default: redacted_<input>)",
        default=None,
    )
    parser.add_argument(
        "--mapping-output",
        help="Path to save mapping JSON (original → placeholder)",
        default=None,
    )
    parser.add_argument(
        "--skip-patterns",
        help="Comma-separated list of pattern names to skip",
        default="",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-redaction verification",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not redact document metadata",
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="List available patterns and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Handle verbose mode
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List patterns and exit
    if args.list_patterns:
        print("\nAvailable redaction patterns:\n")
        for name, config in sorted(REDACTION_PATTERNS.items()):
            print(f"  {name}:")
            print(f"    Description: {config.get('description', 'N/A')}")
            print(f"    Prefix: {config.get('prefix', 'N/A')}")
            print()
        return EXIT_SUCCESS

    # Validate input file exists
    try:
        validate_pdf_file(args.input)
    except FileNotFoundError:
        logger.error(f"File not found: {args.input}")
        return EXIT_FILE_NOT_FOUND
    except PDFValidationError as e:
        logger.error(f"Invalid PDF: {e}")
        return EXIT_INVALID_PDF

    # Generate output paths if not provided
    input_path = Path(args.input)

    if args.output is None:
        args.output = str(input_path.parent / f"redacted_{input_path.name}")

    if args.mapping_output is None:
        output_path = Path(args.output)
        args.mapping_output = str(
            output_path.parent / f"{output_path.stem}_mappings.json"
        )

    # Parse skip patterns
    skip_patterns = set()
    if args.skip_patterns:
        skip_patterns = {p.strip() for p in args.skip_patterns.split(",")}
        logger.info(f"Skipping patterns: {skip_patterns}")

    # Create redactor and process
    try:
        redactor = PDFRedactor(
            skip_patterns=skip_patterns,
            verify_redaction=not args.no_verify,
        )

        stats = redactor.redact_pdf(
            args.input,
            args.output,
            redact_metadata=not args.no_metadata,
        )

        # Save mappings
        redactor.save_mappings(args.mapping_output)

        print(f"\nComplete! Files created:")
        print(f"  - Redacted PDF: {args.output}")
        print(f"  - Mappings: {args.mapping_output}")

        if stats.verification_passed:
            print("  - Verification: PASSED")
        elif not args.no_verify:
            print("  - Verification: WARNINGS (check logs)")

        return EXIT_SUCCESS

    except (PDFValidationError, PDFProcessingError) as e:
        logger.error(str(e))
        return EXIT_PROCESSING_ERROR
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return EXIT_PROCESSING_ERROR


if __name__ == "__main__":
    sys.exit(main())
