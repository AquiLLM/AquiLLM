"""
Local Tesseract OCR provider.
"""

import io
import structlog
from typing import Any, Dict

logger = structlog.stdlib.get_logger(__name__)


def extract_text_with_tesseract(file_content: bytes, convert_to_latex: bool = False) -> Dict[str, Any]:
    """Extract text from image using local Tesseract OCR."""
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception as exc:
        raise ValueError("Local OCR dependencies are missing (Pillow/pytesseract).") from exc

    try:
        image = Image.open(io.BytesIO(file_content))
        extracted_text = (pytesseract.image_to_string(image) or "").strip()
    except Exception as exc:
        raise ValueError(
            "Local OCR failed. Ensure `tesseract-ocr` is installed in the container."
        ) from exc

    if convert_to_latex:
        logger.info("Local OCR provider does not support LaTeX conversion; returning plain extracted text.")

    return {
        "extracted_text": extracted_text or "NO READABLE TEXT",
        "provider": "local",
        "model": "tesseract",
    }


__all__ = ['extract_text_with_tesseract']
