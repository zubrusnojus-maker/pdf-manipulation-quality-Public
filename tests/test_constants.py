"""
Tests for constants.py module.

These tests verify that all configuration constants are properly defined
and have valid values.
"""

import pytest

from constants import (
    COLOR_BLACK,
    COLOR_WHITE,
    DEFAULT_TEMPERATURE,
    DEVICE_CPU,
    DEVICE_GPU,
    DPI_DEFAULT,
    DPI_MAX,
    DPI_MIN,
    EXIT_FILE_NOT_FOUND,
    EXIT_INVALID_ARGUMENTS,
    EXIT_INVALID_PDF,
    EXIT_PROCESSING_ERROR,
    EXIT_SUCCESS,
    LAYOUT_MODEL_LAYOUTLMV3,
    MAPPING_FILE_PERMISSIONS,
    MAPPING_FILE_WARNING,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_INPUT_LENGTH_LARGE,
    MAX_INPUT_LENGTH_SMALL,
    MAX_NEW_TOKENS_LARGE,
    MAX_NEW_TOKENS_SMALL,
    MIN_TEXT_LENGTH,
    OCR_MODEL_TROCR_LARGE,
    PAGE_MARGIN_PX,
    PATTERN_EXCLUSIONS,
    REDACTION_PATTERNS,
    SUPPORTED_PDF_EXTENSIONS,
    TEXT_MODEL_LLAMA,
    TEXT_MODEL_MISTRAL,
    VALID_MODEL_SIZES,
    VALID_OPERATIONS,
)


class TestFileSizeConstants:
    """Tests for file size limit constants."""

    def test_max_file_size_mb_positive(self) -> None:
        """MAX_FILE_SIZE_MB should be positive."""
        assert MAX_FILE_SIZE_MB > 0

    def test_max_file_size_bytes_calculated(self) -> None:
        """MAX_FILE_SIZE_BYTES should equal MB * 1024 * 1024."""
        assert MAX_FILE_SIZE_BYTES == MAX_FILE_SIZE_MB * 1024 * 1024

    def test_max_file_size_reasonable(self) -> None:
        """MAX_FILE_SIZE_MB should be reasonable (10-500 MB)."""
        assert 10 <= MAX_FILE_SIZE_MB <= 500


class TestTextLengthConstants:
    """Tests for text length limit constants."""

    def test_min_text_length_positive(self) -> None:
        """MIN_TEXT_LENGTH should be positive."""
        assert MIN_TEXT_LENGTH > 0

    def test_max_input_lengths_ordered(self) -> None:
        """Large model should have higher limit than small."""
        assert MAX_INPUT_LENGTH_LARGE > MAX_INPUT_LENGTH_SMALL

    def test_max_new_tokens_ordered(self) -> None:
        """Large model should generate more tokens."""
        assert MAX_NEW_TOKENS_LARGE > MAX_NEW_TOKENS_SMALL


class TestDPIConstants:
    """Tests for DPI configuration constants."""

    def test_dpi_default_in_range(self) -> None:
        """DPI_DEFAULT should be between MIN and MAX."""
        assert DPI_MIN <= DPI_DEFAULT <= DPI_MAX

    def test_dpi_min_positive(self) -> None:
        """DPI_MIN should be positive."""
        assert DPI_MIN > 0

    def test_dpi_max_reasonable(self) -> None:
        """DPI_MAX should be reasonable (under 1200)."""
        assert DPI_MAX <= 1200


class TestColorConstants:
    """Tests for color constants."""

    def test_color_black_is_black(self) -> None:
        """COLOR_BLACK should be (0, 0, 0)."""
        assert COLOR_BLACK == (0.0, 0.0, 0.0)

    def test_color_white_is_white(self) -> None:
        """COLOR_WHITE should be (1, 1, 1)."""
        assert COLOR_WHITE == (1.0, 1.0, 1.0)

    def test_color_values_in_range(self) -> None:
        """Color values should be in 0-1 range."""
        for color in [COLOR_BLACK, COLOR_WHITE]:
            for value in color:
                assert 0.0 <= value <= 1.0


class TestDeviceConstants:
    """Tests for device configuration constants."""

    def test_device_gpu_is_zero(self) -> None:
        """DEVICE_GPU should be 0."""
        assert DEVICE_GPU == 0

    def test_device_cpu_is_negative(self) -> None:
        """DEVICE_CPU should be -1."""
        assert DEVICE_CPU == -1


class TestModelConstants:
    """Tests for model name constants."""

    def test_ocr_model_valid(self) -> None:
        """OCR model should be a valid HuggingFace model path."""
        assert "/" in OCR_MODEL_TROCR_LARGE
        assert "trocr" in OCR_MODEL_TROCR_LARGE.lower()

    def test_text_model_mistral_valid(self) -> None:
        """Mistral model should be valid."""
        assert "/" in TEXT_MODEL_MISTRAL
        assert "mistral" in TEXT_MODEL_MISTRAL.lower()

    def test_text_model_llama_valid(self) -> None:
        """Llama model should be valid."""
        assert "/" in TEXT_MODEL_LLAMA
        assert "llama" in TEXT_MODEL_LLAMA.lower()

    def test_layout_model_valid(self) -> None:
        """Layout model should be valid."""
        assert "/" in LAYOUT_MODEL_LAYOUTLMV3


class TestGenerationConstants:
    """Tests for text generation constants."""

    def test_temperature_in_range(self) -> None:
        """Temperature should be between 0 and 2."""
        assert 0.0 <= DEFAULT_TEMPERATURE <= 2.0


class TestExitCodeConstants:
    """Tests for exit code constants."""

    def test_exit_success_is_zero(self) -> None:
        """EXIT_SUCCESS should be 0."""
        assert EXIT_SUCCESS == 0

    def test_exit_codes_unique(self) -> None:
        """All exit codes should be unique."""
        codes = [
            EXIT_SUCCESS,
            EXIT_FILE_NOT_FOUND,
            EXIT_INVALID_PDF,
            EXIT_PROCESSING_ERROR,
            EXIT_INVALID_ARGUMENTS,
        ]
        assert len(codes) == len(set(codes))

    def test_error_codes_non_zero(self) -> None:
        """Error exit codes should be non-zero."""
        assert EXIT_FILE_NOT_FOUND != 0
        assert EXIT_INVALID_PDF != 0
        assert EXIT_PROCESSING_ERROR != 0
        assert EXIT_INVALID_ARGUMENTS != 0


class TestRedactionPatterns:
    """Tests for redaction pattern constants."""

    def test_patterns_not_empty(self) -> None:
        """REDACTION_PATTERNS should contain patterns."""
        assert len(REDACTION_PATTERNS) > 0

    def test_patterns_have_required_keys(self) -> None:
        """Each pattern should have required keys."""
        for name, config in REDACTION_PATTERNS.items():
            assert "pattern" in config, f"Pattern {name} missing 'pattern'"
            assert "prefix" in config, f"Pattern {name} missing 'prefix'"

    def test_patterns_have_descriptions(self) -> None:
        """Each pattern should have a description."""
        for name, config in REDACTION_PATTERNS.items():
            assert "description" in config, f"Pattern {name} missing 'description'"

    def test_common_patterns_present(self) -> None:
        """Common sensitive data patterns should be present."""
        expected_patterns = ["email", "phone_uk", "phone_us", "credit_card", "ssn"]
        for pattern in expected_patterns:
            assert pattern in REDACTION_PATTERNS, f"Missing pattern: {pattern}"

    def test_pattern_exclusions_valid(self) -> None:
        """Pattern exclusions should reference valid patterns."""
        for pattern, exclusions in PATTERN_EXCLUSIONS.items():
            assert pattern in REDACTION_PATTERNS, f"Exclusion for unknown pattern: {pattern}"
            for excluded in exclusions:
                assert excluded in REDACTION_PATTERNS, f"Unknown exclusion: {excluded}"


class TestValidSets:
    """Tests for valid value sets."""

    def test_valid_model_sizes(self) -> None:
        """VALID_MODEL_SIZES should contain expected values."""
        assert "small" in VALID_MODEL_SIZES
        assert "large" in VALID_MODEL_SIZES

    def test_valid_operations(self) -> None:
        """VALID_OPERATIONS should contain expected values."""
        assert "summarize" in VALID_OPERATIONS
        assert "rewrite" in VALID_OPERATIONS

    def test_supported_pdf_extensions(self) -> None:
        """SUPPORTED_PDF_EXTENSIONS should contain .pdf."""
        assert ".pdf" in SUPPORTED_PDF_EXTENSIONS


class TestSecurityConstants:
    """Tests for security-related constants."""

    def test_mapping_file_permissions_restrictive(self) -> None:
        """MAPPING_FILE_PERMISSIONS should be restrictive (owner only)."""
        # 0o600 = owner read/write only
        assert MAPPING_FILE_PERMISSIONS == 0o600

    def test_mapping_file_warning_not_empty(self) -> None:
        """MAPPING_FILE_WARNING should contain warning text."""
        assert len(MAPPING_FILE_WARNING) > 0
        assert "warning" in MAPPING_FILE_WARNING.lower() or "sensitive" in MAPPING_FILE_WARNING.lower()


class TestLayoutConstants:
    """Tests for PDF layout constants."""

    def test_page_margin_reasonable(self) -> None:
        """PAGE_MARGIN_PX should be reasonable (36-144 points)."""
        assert 36 <= PAGE_MARGIN_PX <= 144
