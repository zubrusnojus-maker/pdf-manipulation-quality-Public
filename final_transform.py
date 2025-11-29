#!/usr/bin/env python3
"""
Final Transform Module

A placeholder module for implementing custom text transformations after
OCR extraction and before reinsertion into PDFs.

This module provides an extensible framework for adding custom text
processing pipelines such as:
- Translation
- Grammar correction
- Format conversion
- Custom entity extraction
- Text classification
- Content filtering

Usage:
    from final_transform import FinalTransform, TransformResult

    # Create a transform pipeline
    transform = FinalTransform()

    # Add custom transformers
    transform.add_transformer("uppercase", lambda x: x.upper())

    # Apply transformations
    result = transform.apply("Hello World")
    print(result.text)  # "HELLO WORLD"

Future enhancements can extend this module to support:
- Async transformations for better performance
- Caching of transformation results
- Batch processing of multiple texts
- Integration with external APIs
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    """Result of a text transformation."""

    text: str
    original_text: str
    transformers_applied: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def was_modified(self) -> bool:
        """Check if text was modified."""
        return self.text != self.original_text

    def __str__(self) -> str:
        """Return the transformed text."""
        return self.text


class BaseTransformer(ABC):
    """Abstract base class for text transformers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the transformer name."""
        pass

    @abstractmethod
    def transform(self, text: str) -> str:
        """Transform the input text."""
        pass


class IdentityTransformer(BaseTransformer):
    """A no-op transformer that returns text unchanged."""

    @property
    def name(self) -> str:
        return "identity"

    def transform(self, text: str) -> str:
        return text


class UppercaseTransformer(BaseTransformer):
    """Transform text to uppercase."""

    @property
    def name(self) -> str:
        return "uppercase"

    def transform(self, text: str) -> str:
        return text.upper()


class LowercaseTransformer(BaseTransformer):
    """Transform text to lowercase."""

    @property
    def name(self) -> str:
        return "lowercase"

    def transform(self, text: str) -> str:
        return text.lower()


class TrimTransformer(BaseTransformer):
    """Remove leading and trailing whitespace."""

    @property
    def name(self) -> str:
        return "trim"

    def transform(self, text: str) -> str:
        return text.strip()


class FunctionTransformer(BaseTransformer):
    """Wrapper to create a transformer from a function."""

    def __init__(self, name: str, func: Callable[[str], str]) -> None:
        """
        Initialize with a function.

        Args:
            name: Name for this transformer
            func: Function that takes and returns a string
        """
        self._name = name
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    def transform(self, text: str) -> str:
        return self._func(text)


class FinalTransform:
    """
    Manages a pipeline of text transformations.

    This class provides a flexible way to chain multiple text transformations
    together. Transformations are applied in the order they are added.

    Example:
        transform = FinalTransform()
        transform.add_transformer("trim", lambda x: x.strip())
        transform.add_transformer("upper", lambda x: x.upper())

        result = transform.apply("  hello world  ")
        # result.text == "HELLO WORLD"
    """

    def __init__(self) -> None:
        """Initialize with an empty transformer pipeline."""
        self._transformers: list[BaseTransformer] = []

    def add_transformer(
        self,
        name: str,
        func: Callable[[str], str] | None = None,
        transformer: BaseTransformer | None = None,
    ) -> "FinalTransform":
        """
        Add a transformer to the pipeline.

        Args:
            name: Name for the transformer (used for logging/tracking)
            func: A function that transforms text (alternative to transformer)
            transformer: A BaseTransformer instance (alternative to func)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If neither func nor transformer is provided
        """
        if transformer is not None:
            self._transformers.append(transformer)
        elif func is not None:
            self._transformers.append(FunctionTransformer(name, func))
        else:
            raise ValueError("Either func or transformer must be provided")

        logger.debug(f"Added transformer: {name}")
        return self

    def add_builtin(self, name: str) -> "FinalTransform":
        """
        Add a built-in transformer by name.

        Available built-ins:
        - "identity": No-op (returns text unchanged)
        - "uppercase": Convert to uppercase
        - "lowercase": Convert to lowercase
        - "trim": Remove leading/trailing whitespace

        Args:
            name: Name of the built-in transformer

        Returns:
            Self for method chaining

        Raises:
            ValueError: If built-in name is not recognized
        """
        builtins: dict[str, type[BaseTransformer]] = {
            "identity": IdentityTransformer,
            "uppercase": UppercaseTransformer,
            "lowercase": LowercaseTransformer,
            "trim": TrimTransformer,
        }

        if name not in builtins:
            raise ValueError(
                f"Unknown built-in transformer: {name}. "
                f"Available: {list(builtins.keys())}"
            )

        self._transformers.append(builtins[name]())
        return self

    def remove_transformer(self, name: str) -> bool:
        """
        Remove a transformer by name.

        Args:
            name: Name of the transformer to remove

        Returns:
            True if removed, False if not found
        """
        original_count = len(self._transformers)
        self._transformers = [t for t in self._transformers if t.name != name]
        return len(self._transformers) < original_count

    def clear(self) -> None:
        """Remove all transformers."""
        self._transformers.clear()

    @property
    def transformer_count(self) -> int:
        """Return the number of transformers in the pipeline."""
        return len(self._transformers)

    @property
    def transformer_names(self) -> list[str]:
        """Return the names of all transformers in order."""
        return [t.name for t in self._transformers]

    def apply(self, text: str) -> TransformResult:
        """
        Apply all transformers to the input text.

        Args:
            text: The text to transform

        Returns:
            TransformResult containing the transformed text and metadata
        """
        original = text
        current = text
        applied: list[str] = []

        for transformer in self._transformers:
            try:
                current = transformer.transform(current)
                applied.append(transformer.name)
                logger.debug(f"Applied transformer: {transformer.name}")
            except Exception as e:
                logger.warning(
                    f"Transformer {transformer.name} failed: {e}. Skipping."
                )

        return TransformResult(
            text=current,
            original_text=original,
            transformers_applied=applied,
            metadata={"transformer_count": len(applied)},
        )

    def apply_batch(self, texts: list[str]) -> list[TransformResult]:
        """
        Apply transformers to multiple texts.

        Args:
            texts: List of texts to transform

        Returns:
            List of TransformResult objects
        """
        return [self.apply(text) for text in texts]


# =============================================================================
# Convenience functions for common use cases
# =============================================================================


def create_default_pipeline() -> FinalTransform:
    """
    Create a pipeline with sensible defaults.

    Returns:
        FinalTransform with trim transformer applied
    """
    return FinalTransform().add_builtin("trim")


def create_normalization_pipeline() -> FinalTransform:
    """
    Create a pipeline for text normalization.

    Returns:
        FinalTransform with trim and lowercase transformers
    """
    return FinalTransform().add_builtin("trim").add_builtin("lowercase")


# =============================================================================
# Placeholder for future ML-based transformers
# =============================================================================


class TranslationTransformer(BaseTransformer):
    """
    Placeholder for translation transformer.

    Future implementation will use ML models for translation.
    """

    def __init__(self, source_lang: str = "en", target_lang: str = "es") -> None:
        """
        Initialize translator.

        Args:
            source_lang: Source language code
            target_lang: Target language code
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        logger.warning(
            "TranslationTransformer is a placeholder. "
            "Implement with actual translation model."
        )

    @property
    def name(self) -> str:
        return f"translate_{self.source_lang}_to_{self.target_lang}"

    def transform(self, text: str) -> str:
        # Placeholder: return original text
        # Future: integrate with translation model
        logger.info(f"Translation placeholder: {self.source_lang} -> {self.target_lang}")
        return text


class GrammarCorrectionTransformer(BaseTransformer):
    """
    Placeholder for grammar correction transformer.

    Future implementation will use ML models for grammar correction.
    """

    @property
    def name(self) -> str:
        return "grammar_correction"

    def transform(self, text: str) -> str:
        # Placeholder: return original text
        # Future: integrate with grammar correction model
        logger.info("Grammar correction placeholder")
        return text


# =============================================================================
# CLI Interface (for testing)
# =============================================================================


def main() -> None:
    """CLI entry point for testing transformations."""
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Final Transform Module")
        print()
        print("Usage: python final_transform.py <text>")
        print()
        print("This module provides text transformation pipelines.")
        print("See docstring for API usage.")
        return

    text = " ".join(sys.argv[1:])

    # Demo the transformation pipeline
    pipeline = create_default_pipeline()
    pipeline.add_builtin("uppercase")

    result = pipeline.apply(text)

    print(f"Original:    '{result.original_text}'")
    print(f"Transformed: '{result.text}'")
    print(f"Modified:    {result.was_modified}")
    print(f"Applied:     {result.transformers_applied}")


if __name__ == "__main__":
    main()
