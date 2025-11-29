"""Tests for PDF redaction functionality."""

from pdf_redactor import PDFRedactor


class TestPDFRedactor:
    """Test suite for PDFRedactor class."""

    def test_init(self):
        """Test PDFRedactor initialization."""
        redactor = PDFRedactor()
        assert redactor.mappings == {}
        assert len(redactor.counters) == 0

    def test_detect_monetary_amounts(self):
        """Test detection of monetary amounts."""
        redactor = PDFRedactor()
        text = "The balance is 9,220.51 and pending is 12,020.51"
        detections = redactor.detect_sensitive_data(text)

        assert len(detections) == 2
        assert detections[0][1] == "monetary_amount"
        assert detections[0][0] == "9,220.51"
        assert "Amount_" in detections[0][2]

    def test_detect_account_numbers(self):
        """Test detection of account numbers."""
        redactor = PDFRedactor()
        text = "Account number: 64061866"
        detections = redactor.detect_sensitive_data(text)

        assert len(detections) == 1
        assert detections[0][1] == "account_number"
        assert detections[0][0] == "64061866"

    def test_detect_dates(self):
        """Test detection of dates."""
        redactor = PDFRedactor()
        text = "Date: 30/09/2021 and 01-12-2021"
        detections = redactor.detect_sensitive_data(text)

        assert len(detections) == 2
        assert all(d[1] == "date" for d in detections)

    def test_detect_postcodes(self):
        """Test detection of UK postcodes."""
        redactor = PDFRedactor()
        text = "Address: HA9 6BA"
        detections = redactor.detect_sensitive_data(text)

        postcode_detections = [d for d in detections if d[1] == "postcode"]
        assert len(postcode_detections) >= 1
        assert "HA9 6BA" in [d[0] for d in postcode_detections]

    def test_mapping_consistency(self):
        """Test that same values map to same placeholders."""
        redactor = PDFRedactor()
        text1 = "Amount: 9,220.51"
        text2 = "Balance: 9,220.51"

        detections1 = redactor.detect_sensitive_data(text1)
        detections2 = redactor.detect_sensitive_data(text2)

        assert detections1[0][2] == detections2[0][2]
