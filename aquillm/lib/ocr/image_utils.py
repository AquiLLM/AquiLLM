"""
Image processing utilities for OCR.
"""

import io
import logging
from typing import Any

logger = logging.getLogger(__name__)


def read_image_bytes(image_input: Any) -> bytes:
    """Read image bytes from various input types."""
    import os
    
    try:
        if isinstance(image_input, str) and os.path.exists(image_input):
            with open(image_input, "rb") as f:
                return f.read()

        if isinstance(image_input, bytes):
            return image_input

        if hasattr(image_input, "read"):
            return image_input.read()

        raise ValueError(f"Unsupported image_input type: {type(image_input)}")
    except Exception as exc:
        raise ValueError(f"Could not process image file: {exc}") from exc


def get_image_mime_type(file_content: bytes) -> str:
    """Detect image MIME type from file content."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(file_content)) as image:
            image_format = (image.format or "").lower()
    except Exception:
        return "image/jpeg"

    return {
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "gif": "image/gif",
        "heif": "image/heif",
        "heic": "image/heic",
    }.get(image_format, "image/jpeg")


def resize_image_for_ocr(file_content: bytes, max_dimension: int = 2048, quality: int = 85) -> bytes:
    """Resize large images to fit within vision model context limits.
    
    Large images create massive base64 strings that can exceed the model's
    context window. This function resizes images to a reasonable size while
    preserving readability for OCR.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return file_content

    try:
        with Image.open(io.BytesIO(file_content)) as image:
            original_format = (image.format or "JPEG").upper()
            width, height = image.size
            
            if width <= max_dimension and height <= max_dimension:
                if len(file_content) <= 500_000:
                    return file_content
            
            if width > height:
                if width > max_dimension:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_width, new_height = width, height
            else:
                if height > max_dimension:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                else:
                    new_width, new_height = width, height
            
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            
            if (new_width, new_height) != (width, height):
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug(
                    "Resized image from %dx%d to %dx%d for OCR",
                    width, height, new_width, new_height
                )
            
            output = io.BytesIO()
            save_format = "JPEG" if original_format in ("JPEG", "JPG") else original_format
            if save_format not in ("JPEG", "PNG", "WEBP", "GIF"):
                save_format = "JPEG"
            
            if save_format == "JPEG":
                image.save(output, format=save_format, quality=quality, optimize=True)
            else:
                image.save(output, format=save_format, optimize=True)
            
            return output.getvalue()
    except Exception as exc:
        logger.warning("Failed to resize image for OCR: %s", exc)
        return file_content


__all__ = [
    'read_image_bytes',
    'get_image_mime_type',
    'resize_image_for_ocr',
]
