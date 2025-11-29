"""
Shared utilities for PDF processing toolkit.

This module provides common functionality used across pdf_processor.py,
pdf_redactor.py, and quick_pdf_process.py to avoid code duplication.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import fitz

from constants import (
    COLOR_BLACK,
    COLOR_WHITE,
    DPI_DEFAULT,
    DPI_MAX,
    DPI_MIN,
    EXIT_FILE_NOT_FOUND,
    EXIT_INVALID_PDF,
    MAPPING_FILE_PERMISSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    PAGE_MARGIN_PX,
    SUPPORTED_PDF_EXTENSIONS,
)

# Configure logging
logger = logging.getLogger(__name__)


class PDFValidationError(Exception):
    """Raised when PDF validation fails."""

    pass


class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""

    pass


def setup_logging(
    level: int = logging.INFO,
    log_format: str | None = None,
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        level: Logging level (default: INFO)
        log_format: Custom log format string

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger(__name__)


def validate_pdf_file(
    file_path: str | Path,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
    check_magic_bytes: bool = True,
) -> Path:
    """
    Validate a PDF file before processing.

    Args:
        file_path: Path to the PDF file
        max_size_bytes: Maximum allowed file size in bytes
        check_magic_bytes: Whether to verify PDF magic bytes

    Returns:
        Resolved Path object

    Raises:
        PDFValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path).resolve()

    # Check file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise PDFValidationError(f"Not a file: {path}")

    # Check extension
    if path.suffix.lower() not in SUPPORTED_PDF_EXTENSIONS:
        raise PDFValidationError(
            f"Invalid file extension: {path.suffix}. Expected: {SUPPORTED_PDF_EXTENSIONS}"
        )

    # Check file size
    file_size = path.stat().st_size
    if file_size > max_size_bytes:
        raise PDFValidationError(
            f"File too large: {file_size / (1024 * 1024):.1f}MB. "
            f"Maximum allowed: {MAX_FILE_SIZE_MB}MB"
        )

    if file_size == 0:
        raise PDFValidationError("File is empty")

    # Check magic bytes
    if check_magic_bytes:
        with open(path, "rb") as f:
            magic_bytes = f.read(5)
            if not magic_bytes.startswith(b"%PDF-"):
                raise PDFValidationError(
                    "Invalid PDF format: file does not start with PDF magic bytes"
                )

    return path


def validate_output_path(
    output_path: str | Path,
    input_path: str | Path | None = None,
    allow_overwrite: bool = True,
) -> Path:
    """
    Validate and sanitize output file path.

    Args:
        output_path: Desired output path
        input_path: Original input path (for directory validation)
        allow_overwrite: Whether to allow overwriting existing files

    Returns:
        Sanitized Path object

    Raises:
        PDFValidationError: If path is invalid or unsafe
    """
    path = Path(output_path).resolve()

    # Prevent path traversal attacks
    if ".." in str(output_path):
        raise PDFValidationError("Path traversal detected in output path")

    # Ensure parent directory exists
    if not path.parent.exists():
        raise PDFValidationError(f"Output directory does not exist: {path.parent}")

    # Check for overwrite
    if not allow_overwrite and path.exists():
        raise PDFValidationError(f"Output file already exists: {path}")

    # If input_path provided, validate output is in same or child directory
    if input_path is not None:
        input_dir = Path(input_path).resolve().parent
        try:
            path.relative_to(input_dir)
        except ValueError:
            # Output is not under input directory - this is OK but log a warning
            logger.warning(
                f"Output path {path} is outside input directory {input_dir}"
            )

    return path


def validate_dpi(dpi: int) -> int:
    """
    Validate DPI value for PDF rasterization.

    Args:
        dpi: DPI value to validate

    Returns:
        Validated DPI value

    Raises:
        ValueError: If DPI is out of valid range
    """
    if not isinstance(dpi, int):
        raise ValueError(f"DPI must be an integer, got {type(dpi).__name__}")

    if dpi < DPI_MIN or dpi > DPI_MAX:
        raise ValueError(f"DPI must be between {DPI_MIN} and {DPI_MAX}, got {dpi}")

    return dpi


def generate_output_path(
    input_path: str | Path,
    prefix: str = "edited_",
    suffix: str = "",
) -> Path:
    """
    Generate output path from input path.

    Args:
        input_path: Original input file path
        prefix: Prefix to add to filename
        suffix: Suffix to add before extension

    Returns:
        Generated output Path
    """
    input_path = Path(input_path)
    stem = input_path.stem
    extension = input_path.suffix

    output_name = f"{prefix}{stem}{suffix}{extension}"
    return input_path.parent / output_name


def set_secure_file_permissions(file_path: str | Path) -> None:
    """
    Set restrictive permissions on a file (owner read/write only).

    Args:
        file_path: Path to the file

    Note:
        On Windows, this has limited effect.
    """
    path = Path(file_path)
    if not path.exists():
        return

    try:
        os.chmod(path, MAPPING_FILE_PERMISSIONS)
        logger.debug(f"Set secure permissions on {path}")
    except OSError as e:
        logger.warning(f"Could not set secure permissions on {path}: {e}")


def int_to_rgb(color_int: int) -> tuple[float, float, float]:
    """
    Convert integer color to RGB tuple (0-1 range).

    Args:
        color_int: Integer color value (e.g., 0x000000 for black)

    Returns:
        RGB tuple with values in 0-1 range
    """
    if color_int == 0:
        return COLOR_BLACK

    if color_int < 0:
        return COLOR_BLACK

    r = ((color_int >> 16) & 0xFF) / 255.0
    g = ((color_int >> 8) & 0xFF) / 255.0
    b = (color_int & 0xFF) / 255.0
    return (r, g, b)


def get_page_text_rect(
    page: "fitz.Page",
    margin: int = PAGE_MARGIN_PX,
) -> Any:
    """
    Get a rectangle for text insertion with margins.

    Args:
        page: PyMuPDF page object
        margin: Margin in points (default 72 = 1 inch)

    Returns:
        fitz.Rect object for text area
    """
    import fitz

    return fitz.Rect(
        margin,
        margin,
        page.rect.width - margin,
        page.rect.height - margin,
    )


def truncate_text(
    text: str,
    max_length: int,
    add_ellipsis: bool = False,
) -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        add_ellipsis: Whether to add "..." at end

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    if add_ellipsis and max_length > 3:
        return text[: max_length - 3] + "..."

    return text[:max_length]


def cleanup_resources(*resources: Any) -> None:
    """
    Safely close/cleanup multiple resources.

    Args:
        *resources: Objects with .close() method to cleanup
    """
    for resource in resources:
        if resource is None:
            continue
        try:
            if hasattr(resource, "close"):
                resource.close()
        except Exception as e:
            logger.warning(f"Error closing resource {type(resource).__name__}: {e}")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def check_gpu_available() -> bool:
    """
    Check if GPU is available for model inference.

    Returns:
        True if CUDA GPU is available
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> int:
    """
    Get the appropriate device for model inference.

    Returns:
        0 for GPU, -1 for CPU
    """
    return 0 if check_gpu_available() else -1


def safe_exit(code: int, message: str | None = None) -> None:
    """
    Exit the program safely with optional message.

    Args:
        code: Exit code
        message: Optional message to print
    """
    if message:
        if code == 0:
            logger.info(message)
        else:
            logger.error(message)
    sys.exit(code)
