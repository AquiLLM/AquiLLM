"""
OCR providers and utilities.

Provides:
- Tesseract (local) OCR
- Qwen vision model OCR
- Google Gemini OCR
- Auto-selection with fallback chain

For Django integration with API usage logging, use aquillm.ocr_utils.
"""

import structlog
from typing import Any, Dict

from .config import get_ocr_provider, set_usage_logger
from .image_utils import read_image_bytes, get_image_mime_type, resize_image_for_ocr
from .tesseract import extract_text_with_tesseract
from .qwen import extract_text_with_qwen
from .gemini import extract_text_with_gemini, GeminiCostTracker, cost_tracker

logger = structlog.stdlib.get_logger(__name__)


def extract_text_from_image(image_input: Any, convert_to_latex: bool = False) -> Dict[str, Any]:
    """
    Extract text from image using configured OCR provider.
    
    APP_OCR_PROVIDER controls OCR backend:
      - auto (default): Qwen OCR first, then local OCR, then Gemini fallback
      - qwen: dedicated vllm_ocr Qwen vision OCR service only
      - local: local OCR only
      - gemini: Gemini OCR only
    """
    file_content = read_image_bytes(image_input)
    provider = get_ocr_provider()

    errors: list[str] = []
    if provider in {"auto", "qwen"}:
        try:
            return extract_text_with_qwen(file_content, convert_to_latex=convert_to_latex)
        except Exception as exc:
            errors.append(f"qwen: {exc}")
            if provider == "qwen":
                raise ValueError(f"OCR processing failed: {exc}") from exc
            logger.warning("Qwen OCR failed; trying local OCR fallback. Error: %s", exc)

    if provider in {"auto", "local"}:
        try:
            return extract_text_with_tesseract(file_content, convert_to_latex=convert_to_latex)
        except Exception as exc:
            errors.append(f"local: {exc}")
            if provider == "local":
                raise ValueError(f"OCR processing failed: {exc}") from exc
            logger.warning("Local OCR failed; trying Gemini OCR fallback. Error: %s", exc)

    if provider in {"auto", "gemini"}:
        try:
            return extract_text_with_gemini(file_content, convert_to_latex=convert_to_latex)
        except Exception as exc:
            errors.append(f"gemini: {exc}")

    raise ValueError(f"OCR processing failed: {' | '.join(errors)}")


__all__ = [
    # Main function
    'extract_text_from_image',
    # Providers
    'extract_text_with_tesseract',
    'extract_text_with_qwen',
    'extract_text_with_gemini',
    # Cost tracking
    'GeminiCostTracker',
    'cost_tracker',
    # Config
    'set_usage_logger',
    # Image utils
    'read_image_bytes',
    'get_image_mime_type',
    'resize_image_for_ocr',
]
