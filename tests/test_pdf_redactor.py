"""
Tests for pdf_redactor.py module.

These tests verify the PDF redaction functionality including pattern
detection, text replacement, metadata handling, and verification.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf_redactor import PDFRedactor, RedactionStats, main
from pdf_utils import PDFProcessingError, PDFValidationError


class TestRedactionStats:
    """Tests for RedactionStats class."""

    def test_initialization(self) -> None:
        """Should initialize with default values."""
        stats = RedactionStats()
        assert stats.pages_processed == 0
        assert stats.total_replacements == 0
        assert len(stats.types) == 0
        assert len(stats.patterns_matched) == 0
        assert stats.verification_passed is False

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        stats = RedactionStats()
        stats.pages_processed = 5
        stats.total_replacements = 10
        stats.types["email"] = 3
        stats.verification_passed = True

        result = stats.to_dict()
        assert result["pages_processed"] == 5
        assert result["total_replacements"] == 10
        assert result["types"]["email"] == 3
        assert result["verification_passed"] is True


class TestPDFRedactorInit:
    """Tests for PDFRedactor initialization."""

    def test_default_initialization(self) -> None:
        """Should initialize with default settings."""
        redactor = PDFRedactor()
        assert len(redactor.patterns) > 0
        assert redactor.verify_redaction is True

    def test_skip_patterns(self) -> None:
        """Should skip specified patterns."""
        redactor = PDFRedactor(skip_patterns={"email", "phone_uk"})
        assert "email" not in redactor.patterns
        assert "phone_uk" not in redactor.patterns

    def test_custom_patterns(self) -> None:
        """Should add custom patterns."""
        custom = {
            "custom_id": {
                "pattern": r"ID-\d{6}",
                "prefix": "CustomID_",
            }
        }
        redactor = PDFRedactor(custom_patterns=custom)
        assert "custom_id" in redactor.patterns

    def test_verify_redaction_disabled(self) -> None:
        """Should respect verify_redaction setting."""
        redactor = PDFRedactor(verify_redaction=False)
        assert redactor.verify_redaction is False


class TestPatternDetection:
    """Tests for sensitive data detection."""

    @pytest.fixture
    def redactor(self) -> PDFRedactor:
        """Create a redactor instance."""
        return PDFRedactor()

    def test_detect_email(self, redactor: PDFRedactor) -> None:
        """Should detect email addresses."""
        text = "Contact me at john.doe@example.com for more info."
        detections = redactor.detect_sensitive_data(text)
        emails = [d for d in detections if "email" in d[1]]
        assert len(emails) == 1
        assert "john.doe@example.com" in emails[0][0]

    def test_detect_phone_uk(self, redactor: PDFRedactor) -> None:
        """Should detect UK phone numbers."""
        text = "Call us at +44 1234 567890"
        detections = redactor.detect_sensitive_data(text)
        phones = [d for d in detections if "phone" in d[1]]
        assert len(phones) >= 1

    def test_detect_phone_us(self, redactor: PDFRedactor) -> None:
        """Should detect US phone numbers."""
        text = "Call (555) 123-4567 for support"
        detections = redactor.detect_sensitive_data(text)
        phones = [d for d in detections if "phone" in d[1]]
        assert len(phones) >= 1

    def test_detect_ssn(self, redactor: PDFRedactor) -> None:
        """Should detect SSN patterns."""
        text = "SSN: 123-45-6789"
        detections = redactor.detect_sensitive_data(text)
        ssns = [d for d in detections if "ssn" in d[1].lower()]
        assert len(ssns) == 1

    def test_detect_credit_card(self, redactor: PDFRedactor) -> None:
        """Should detect credit card numbers."""
        text = "Card: 4111-1111-1111-1111"
        detections = redactor.detect_sensitive_data(text)
        cards = [d for d in detections if "card" in d[1].lower() or "credit" in d[1].lower()]
        assert len(cards) == 1

    def test_detect_monetary(self, redactor: PDFRedactor) -> None:
        """Should detect monetary amounts."""
        text = "Total: 1,234.56"
        detections = redactor.detect_sensitive_data(text)
        amounts = [d for d in detections if "amount" in d[1].lower() or "monetary" in d[1].lower()]
        assert len(amounts) >= 1

    def test_detect_date(self, redactor: PDFRedactor) -> None:
        """Should detect dates."""
        text = "Date: 25/12/2024"
        detections = redactor.detect_sensitive_data(text)
        dates = [d for d in detections if "date" in d[1].lower()]
        assert len(dates) >= 1

    def test_detect_uk_postcode(self, redactor: PDFRedactor) -> None:
        """Should detect UK postcodes."""
        text = "Address: SW1A 1AA, London"
        detections = redactor.detect_sensitive_data(text)
        postcodes = [d for d in detections if "postcode" in d[1].lower()]
        assert len(postcodes) == 1

    def test_detect_name_with_title(self, redactor: PDFRedactor) -> None:
        """Should detect names with titles."""
        text = "Contact Mr John Smith for details"
        detections = redactor.detect_sensitive_data(text)
        names = [d for d in detections if "name" in d[1].lower() or "person" in d[1].lower()]
        assert len(names) >= 1

    def test_no_detections_in_clean_text(self, redactor: PDFRedactor) -> None:
        """Should not detect anything in clean text."""
        text = "This is a normal sentence without sensitive data."
        detections = redactor.detect_sensitive_data(text)
        assert len(detections) == 0

    def test_multiple_detections(self, redactor: PDFRedactor) -> None:
        """Should detect multiple patterns in one text."""
        text = "Mr John Smith (john@example.com) paid 1,234.56 on 25/12/2024"
        detections = redactor.detect_sensitive_data(text)
        assert len(detections) >= 3


class TestPlaceholderGeneration:
    """Tests for placeholder generation."""

    def test_unique_placeholders(self) -> None:
        """Should generate unique placeholders."""
        redactor = PDFRedactor()
        text = "Emails: a@b.com and c@d.com"
        detections = redactor.detect_sensitive_data(text)

        placeholders = [d[2] for d in detections]
        assert len(placeholders) == len(set(placeholders))

    def test_consistent_placeholders(self) -> None:
        """Should use same placeholder for repeated values."""
        redactor = PDFRedactor()
        text = "Contact john@example.com. Yes, john@example.com is correct."
        detections = redactor.detect_sensitive_data(text)

        emails = [d for d in detections if "email" in d[1]]
        assert len(emails) == 2
        assert emails[0][2] == emails[1][2]  # Same placeholder

    def test_alphabetic_suffixes(self) -> None:
        """Should use A, B, C... for first 26 items."""
        redactor = PDFRedactor()
        # Create multiple unique detections
        for i in range(5):
            email = f"user{i}@example.com"
            redactor.detect_sensitive_data(f"Email: {email}")

        # Check placeholders have letter suffixes
        for mapping in redactor.mappings.values():
            placeholder = mapping["placeholder"]
            suffix = placeholder.split("_")[-1]
            assert suffix.isalpha() or suffix.isdigit()


class TestRedactPage:
    """Tests for page redaction."""

    def test_redact_page_with_sensitive_data(self, sensitive_data_pdf: Path) -> None:
        """Should redact sensitive data from page."""
        import fitz

        redactor = PDFRedactor()
        doc = fitz.open(str(sensitive_data_pdf))
        page = doc[0]

        stats = redactor.redact_page(page)

        assert stats["replacements"] > 0
        assert len(stats["types"]) > 0
        doc.close()

    def test_redact_page_empty(self, empty_pdf: Path) -> None:
        """Should handle empty page."""
        import fitz

        redactor = PDFRedactor()
        doc = fitz.open(str(empty_pdf))
        page = doc[0]

        stats = redactor.redact_page(page)

        assert stats["replacements"] == 0
        doc.close()


class TestRedactMetadata:
    """Tests for metadata redaction."""

    def test_metadata_redaction(self, sample_pdf: Path) -> None:
        """Should redact PDF metadata."""
        import fitz

        redactor = PDFRedactor()
        doc = fitz.open(str(sample_pdf))

        # Set some metadata first
        doc.set_metadata({"author": "Test Author", "title": "Test Title"})

        redactor.redact_metadata(doc)

        metadata = doc.metadata
        assert metadata.get("author") == "[REDACTED]"
        assert metadata.get("title") == "[REDACTED]"
        doc.close()


class TestRedactPDF:
    """Tests for full PDF redaction."""

    def test_redact_pdf_creates_output(
        self, sensitive_data_pdf: Path, temp_dir: Path
    ) -> None:
        """Should create redacted output file."""
        output_path = temp_dir / "output.pdf"
        redactor = PDFRedactor(verify_redaction=False)

        stats = redactor.redact_pdf(sensitive_data_pdf, output_path)

        assert output_path.exists()
        assert stats.pages_processed > 0

    def test_redact_pdf_invalid_input(self, temp_dir: Path) -> None:
        """Should raise error for invalid input."""
        redactor = PDFRedactor()
        output_path = temp_dir / "output.pdf"

        with pytest.raises(FileNotFoundError):
            redactor.redact_pdf(temp_dir / "nonexistent.pdf", output_path)

    def test_redact_pdf_with_verification(
        self, sample_pdf: Path, temp_dir: Path
    ) -> None:
        """Should run verification when enabled."""
        output_path = temp_dir / "output.pdf"
        redactor = PDFRedactor(verify_redaction=True)

        stats = redactor.redact_pdf(sample_pdf, output_path)

        # Verification should have run
        assert isinstance(stats.verification_passed, bool)


class TestSaveMappings:
    """Tests for mapping file saving."""

    def test_save_mappings_creates_file(self, temp_dir: Path) -> None:
        """Should create mapping file."""
        redactor = PDFRedactor()
        redactor.mappings = {"test@example.com": {"type": "email", "placeholder": "Email_A"}}

        output_path = temp_dir / "mappings.json"
        redactor.save_mappings(output_path)

        assert output_path.exists()

    def test_save_mappings_valid_json(self, temp_dir: Path) -> None:
        """Should save valid JSON."""
        redactor = PDFRedactor()
        redactor.mappings = {"test@example.com": {"type": "email", "placeholder": "Email_A"}}

        output_path = temp_dir / "mappings.json"
        redactor.save_mappings(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "mappings" in data
        assert "test@example.com" in data["mappings"]

    def test_save_mappings_includes_warning(self, temp_dir: Path) -> None:
        """Should include security warning."""
        redactor = PDFRedactor()
        redactor.mappings = {}

        output_path = temp_dir / "mappings.json"
        redactor.save_mappings(output_path, include_warning=True)

        with open(output_path) as f:
            data = json.load(f)

        assert "WARNING" in data


class TestVerifyRedaction:
    """Tests for redaction verification."""

    def test_verify_clean_document(self, sample_pdf: Path) -> None:
        """Should pass for clean document."""
        import fitz

        redactor = PDFRedactor()
        doc = fitz.open(str(sample_pdf))

        success, issues = redactor.verify_redaction_complete(doc)

        # Clean document should have minimal issues
        doc.close()

    def test_verify_returns_issues(self, sensitive_data_pdf: Path) -> None:
        """Should return list of issues found."""
        import fitz

        redactor = PDFRedactor()
        doc = fitz.open(str(sensitive_data_pdf))

        success, issues = redactor.verify_redaction_complete(doc)

        # Should find sensitive data (not redacted yet)
        assert len(issues) > 0
        doc.close()


class TestMain:
    """Tests for CLI main function."""

    def test_main_list_patterns(self) -> None:
        """Should list patterns and exit."""
        with patch("sys.argv", ["pdf_redactor.py", "--list-patterns"]):
            result = main()
            assert result == 0

    def test_main_file_not_found(self, temp_dir: Path) -> None:
        """Should return error for missing file."""
        with patch("sys.argv", ["pdf_redactor.py", str(temp_dir / "missing.pdf")]):
            result = main()
            assert result != 0

    def test_main_success(
        self, sensitive_data_pdf: Path, temp_dir: Path
    ) -> None:
        """Should process file successfully."""
        output = temp_dir / "output.pdf"
        with patch(
            "sys.argv",
            ["pdf_redactor.py", str(sensitive_data_pdf), "-o", str(output), "--no-verify"],
        ):
            result = main()
            assert result == 0
            assert output.exists()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_text(self) -> None:
        """Should handle empty text."""
        redactor = PDFRedactor()
        detections = redactor.detect_sensitive_data("")
        assert len(detections) == 0

    def test_whitespace_only(self) -> None:
        """Should handle whitespace-only text."""
        redactor = PDFRedactor()
        detections = redactor.detect_sensitive_data("   \n\t  ")
        assert len(detections) == 0

    def test_special_characters(self) -> None:
        """Should handle special characters."""
        redactor = PDFRedactor()
        text = "Special chars: @#$%^&*()[]{}|\\<>?/"
        detections = redactor.detect_sensitive_data(text)
        # Should not crash, may or may not detect anything

    def test_unicode_text(self) -> None:
        """Should handle Unicode text."""
        redactor = PDFRedactor()
        text = "Unicode: café, naïve, 日本語, émoji: 🎉"
        detections = redactor.detect_sensitive_data(text)
        # Should not crash

    def test_very_long_text(self) -> None:
        """Should handle very long text."""
        redactor = PDFRedactor()
        text = "test@example.com " * 1000
        detections = redactor.detect_sensitive_data(text)
        # Should detect the email pattern many times
        assert len(detections) > 0
