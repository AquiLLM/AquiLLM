"""
OCR utilities with Django model integration.

This module provides the main OCR interface for the application,
integrating lib/ocr with Django models for API usage logging.
"""

from typing import Any, Dict

from lib.ocr import (
    extract_text_from_image,
    extract_text_with_tesseract,
    extract_text_with_qwen,
    extract_text_with_gemini,
    GeminiCostTracker,
    cost_tracker,
    set_usage_logger,
    read_image_bytes,
    get_image_mime_type,
    resize_image_for_ocr,
)

# Set up Django model integration for usage logging
def _django_usage_logger(operation_type: str, input_tokens: int, output_tokens: int) -> None:
    """Log usage to Django GeminiAPIUsage model."""
    from .models import GeminiAPIUsage
    GeminiAPIUsage.log_usage(
        operation_type=operation_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

# Register the Django logger
set_usage_logger(_django_usage_logger)


def get_gemini_cost_stats() -> Dict[str, Any]:
    """Get Gemini API cost statistics from database."""
    from .models import GeminiAPIUsage

    stats = GeminiAPIUsage.get_total_stats()
    return {
        "total_cost_usd": stats["total_cost"] or 0,
        "input_tokens": stats["total_input_tokens"] or 0,
        "output_tokens": stats["total_output_tokens"] or 0,
        "api_calls": stats["api_calls"] or 0,
    }


__all__ = [
    'extract_text_from_image',
    'extract_text_with_tesseract',
    'extract_text_with_qwen',
    'extract_text_with_gemini',
    'get_gemini_cost_stats',
    'GeminiCostTracker',
    'cost_tracker',
    'read_image_bytes',
    'get_image_mime_type',
    'resize_image_for_ocr',
]
