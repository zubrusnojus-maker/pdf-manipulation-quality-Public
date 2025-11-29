"""
Tests for pdf_utils.py module.

These tests verify the shared utility functions for PDF validation,
path handling, and resource management.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf_utils import (
    PDFProcessingError,
    PDFValidationError,
    check_gpu_available,
    cleanup_resources,
    format_file_size,
    generate_output_path,
    get_device,
    int_to_rgb,
    set_secure_file_permissions,
    setup_logging,
    truncate_text,
    validate_dpi,
    validate_output_path,
    validate_pdf_file,
)


class TestValidatePDFFile:
    """Tests for validate_pdf_file function."""

    def test_valid_pdf(self, sample_pdf: Path) -> None:
        """Should return path for valid PDF."""
        result = validate_pdf_file(sample_pdf)
        assert result == sample_pdf.resolve()

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            validate_pdf_file(temp_dir / "nonexistent.pdf")

    def test_invalid_extension(self, temp_dir: Path) -> None:
        """Should raise PDFValidationError for non-PDF extension."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test")
        with pytest.raises(PDFValidationError, match="Invalid file extension"):
            validate_pdf_file(txt_file)

    def test_invalid_magic_bytes(self, invalid_pdf: Path) -> None:
        """Should raise PDFValidationError for invalid PDF content."""
        with pytest.raises(PDFValidationError, match="Invalid PDF format"):
            validate_pdf_file(invalid_pdf)

    def test_empty_file(self, temp_dir: Path) -> None:
        """Should raise PDFValidationError for empty file."""
        empty_file = temp_dir / "empty.pdf"
        empty_file.touch()
        with pytest.raises(PDFValidationError, match="empty"):
            validate_pdf_file(empty_file)

    def test_directory_not_file(self, temp_dir: Path) -> None:
        """Should raise PDFValidationError for directory."""
        with pytest.raises(PDFValidationError, match="Not a file"):
            validate_pdf_file(temp_dir)

    def test_file_too_large(self, temp_dir: Path) -> None:
        """Should raise PDFValidationError for oversized file."""
        with pytest.raises(PDFValidationError, match="too large"):
            # Pass a very small max_size to trigger the error
            pdf = temp_dir / "test.pdf"
            pdf.write_bytes(b"%PDF-1.4" + b"x" * 100)
            validate_pdf_file(pdf, max_size_bytes=50)


class TestValidateOutputPath:
    """Tests for validate_output_path function."""

    def test_valid_output_path(self, temp_dir: Path) -> None:
        """Should return path for valid output location."""
        output = temp_dir / "output.pdf"
        result = validate_output_path(output)
        assert result == output.resolve()

    def test_path_traversal_rejected(self, temp_dir: Path) -> None:
        """Should reject path traversal attempts."""
        with pytest.raises(PDFValidationError, match="traversal"):
            validate_output_path("../../../etc/passwd")

    def test_parent_directory_missing(self, temp_dir: Path) -> None:
        """Should raise error if parent directory doesn't exist."""
        with pytest.raises(PDFValidationError, match="does not exist"):
            validate_output_path(temp_dir / "nonexistent" / "output.pdf")

    def test_overwrite_not_allowed(self, sample_pdf: Path, temp_dir: Path) -> None:
        """Should raise error when overwrite not allowed."""
        with pytest.raises(PDFValidationError, match="already exists"):
            validate_output_path(sample_pdf, allow_overwrite=False)


class TestValidateDPI:
    """Tests for validate_dpi function."""

    def test_valid_dpi(self) -> None:
        """Should return valid DPI values."""
        assert validate_dpi(150) == 150
        assert validate_dpi(300) == 300

    def test_dpi_too_low(self) -> None:
        """Should reject DPI below minimum."""
        with pytest.raises(ValueError):
            validate_dpi(50)

    def test_dpi_too_high(self) -> None:
        """Should reject DPI above maximum."""
        with pytest.raises(ValueError):
            validate_dpi(1000)

    def test_dpi_not_integer(self) -> None:
        """Should reject non-integer DPI."""
        with pytest.raises(ValueError):
            validate_dpi(300.5)  # type: ignore


class TestGenerateOutputPath:
    """Tests for generate_output_path function."""

    def test_default_prefix(self, temp_dir: Path) -> None:
        """Should add default prefix."""
        input_path = temp_dir / "document.pdf"
        result = generate_output_path(input_path)
        assert result.name == "edited_document.pdf"

    def test_custom_prefix(self, temp_dir: Path) -> None:
        """Should use custom prefix."""
        input_path = temp_dir / "document.pdf"
        result = generate_output_path(input_path, prefix="redacted_")
        assert result.name == "redacted_document.pdf"

    def test_custom_suffix(self, temp_dir: Path) -> None:
        """Should add custom suffix."""
        input_path = temp_dir / "document.pdf"
        result = generate_output_path(input_path, prefix="", suffix="_v2")
        assert result.name == "document_v2.pdf"

    def test_preserves_parent_directory(self, temp_dir: Path) -> None:
        """Should preserve parent directory."""
        input_path = temp_dir / "document.pdf"
        result = generate_output_path(input_path)
        assert result.parent == temp_dir


class TestIntToRGB:
    """Tests for int_to_rgb function."""

    def test_black(self) -> None:
        """Should convert 0 to black."""
        assert int_to_rgb(0) == (0.0, 0.0, 0.0)

    def test_white(self) -> None:
        """Should convert white integer to white RGB."""
        white_int = 0xFFFFFF
        r, g, b = int_to_rgb(white_int)
        assert r == pytest.approx(1.0)
        assert g == pytest.approx(1.0)
        assert b == pytest.approx(1.0)

    def test_red(self) -> None:
        """Should convert red integer correctly."""
        red_int = 0xFF0000
        r, g, b = int_to_rgb(red_int)
        assert r == pytest.approx(1.0)
        assert g == pytest.approx(0.0)
        assert b == pytest.approx(0.0)

    def test_negative_returns_black(self) -> None:
        """Should return black for negative values."""
        assert int_to_rgb(-1) == (0.0, 0.0, 0.0)


class TestTruncateText:
    """Tests for truncate_text function."""

    def test_short_text_unchanged(self) -> None:
        """Should not truncate short text."""
        text = "Hello"
        assert truncate_text(text, 100) == text

    def test_long_text_truncated(self) -> None:
        """Should truncate long text."""
        text = "Hello World"
        result = truncate_text(text, 5)
        assert result == "Hello"
        assert len(result) == 5

    def test_with_ellipsis(self) -> None:
        """Should add ellipsis when requested."""
        text = "Hello World"
        result = truncate_text(text, 8, add_ellipsis=True)
        assert result == "Hello..."
        assert len(result) == 8

    def test_exact_length_unchanged(self) -> None:
        """Should not truncate text at exact max length."""
        text = "Hello"
        assert truncate_text(text, 5) == text


class TestFormatFileSize:
    """Tests for format_file_size function."""

    def test_bytes(self) -> None:
        """Should format bytes correctly."""
        assert "B" in format_file_size(100)

    def test_kilobytes(self) -> None:
        """Should format KB correctly."""
        assert "KB" in format_file_size(2048)

    def test_megabytes(self) -> None:
        """Should format MB correctly."""
        assert "MB" in format_file_size(2 * 1024 * 1024)

    def test_gigabytes(self) -> None:
        """Should format GB correctly."""
        assert "GB" in format_file_size(2 * 1024 * 1024 * 1024)


class TestCleanupResources:
    """Tests for cleanup_resources function."""

    def test_closes_resources(self) -> None:
        """Should call close on resources."""
        mock_resource = MagicMock()
        cleanup_resources(mock_resource)
        mock_resource.close.assert_called_once()

    def test_handles_none(self) -> None:
        """Should handle None resources."""
        cleanup_resources(None)  # Should not raise

    def test_handles_multiple_resources(self) -> None:
        """Should close multiple resources."""
        mock1 = MagicMock()
        mock2 = MagicMock()
        cleanup_resources(mock1, mock2)
        mock1.close.assert_called_once()
        mock2.close.assert_called_once()

    def test_continues_after_error(self) -> None:
        """Should continue closing resources after error."""
        mock1 = MagicMock()
        mock1.close.side_effect = Exception("Error")
        mock2 = MagicMock()
        cleanup_resources(mock1, mock2)
        mock2.close.assert_called_once()


class TestCheckGPUAvailable:
    """Tests for check_gpu_available function."""

    def test_returns_bool(self, mock_torch_cuda: MagicMock) -> None:
        """Should return boolean."""
        result = check_gpu_available()
        assert isinstance(result, bool)

    def test_cpu_mode(self, mock_torch_cuda: MagicMock) -> None:
        """Should return False when CUDA unavailable."""
        assert check_gpu_available() is False


class TestGetDevice:
    """Tests for get_device function."""

    def test_cpu_device(self, mock_torch_cuda: MagicMock) -> None:
        """Should return -1 for CPU."""
        assert get_device() == -1

    def test_gpu_device(self, mock_torch_cuda_available: MagicMock) -> None:
        """Should return 0 for GPU."""
        assert get_device() == 0


class TestSetSecureFilePermissions:
    """Tests for set_secure_file_permissions function."""

    def test_sets_permissions(self, temp_dir: Path) -> None:
        """Should set restrictive permissions."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        set_secure_file_permissions(test_file)
        # Check permissions (owner read/write only)
        mode = test_file.stat().st_mode & 0o777
        assert mode == 0o600

    def test_handles_missing_file(self, temp_dir: Path) -> None:
        """Should handle missing file gracefully."""
        set_secure_file_permissions(temp_dir / "nonexistent.txt")  # Should not raise


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self) -> None:
        """Should return a logger instance."""
        import logging
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)


class TestExceptions:
    """Tests for custom exception classes."""

    def test_pdf_validation_error(self) -> None:
        """Should be able to raise PDFValidationError."""
        with pytest.raises(PDFValidationError):
            raise PDFValidationError("test error")

    def test_pdf_processing_error(self) -> None:
        """Should be able to raise PDFProcessingError."""
        with pytest.raises(PDFProcessingError):
            raise PDFProcessingError("test error")

    def test_exception_messages(self) -> None:
        """Should preserve error messages."""
        try:
            raise PDFValidationError("custom message")
        except PDFValidationError as e:
            assert "custom message" in str(e)
