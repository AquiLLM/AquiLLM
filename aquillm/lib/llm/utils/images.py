"""Resize image data URLs for LLM context limits."""
from __future__ import annotations

import base64
import io
import structlog
from typing import Optional

logger = structlog.stdlib.get_logger(__name__)


def resize_image_data_url_for_llm(
    image_data_url: str,
    *,
    max_dimension: int,
    max_bytes: int,
) -> Optional[str]:
    """
    Resize an image data URL to fit within LLM context limits.

    Returns a smaller base64 data URL or None if resizing fails.
    """
    if not image_data_url or not image_data_url.startswith("data:"):
        return None

    try:
        from PIL import Image

        header, b64_data = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(b64_data)

        if len(image_bytes) <= max_bytes:
            return image_data_url

        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            width, height = img.size

            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            for quality in (70, 50, 35, 20):
                output = io.BytesIO()
                img.save(output, format="JPEG", quality=quality, optimize=True)
                result_bytes = output.getvalue()

                if len(result_bytes) <= max_bytes:
                    encoded = base64.b64encode(result_bytes).decode("ascii")
                    logger.debug(
                        "Resized image for LLM context: %d -> %d bytes (quality=%d)",
                        len(image_bytes),
                        len(result_bytes),
                        quality,
                    )
                    return f"data:image/jpeg;base64,{encoded}"

            img = img.resize((256, int(256 * height / width)), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=20, optimize=True)
            result_bytes = output.getvalue()
            encoded = base64.b64encode(result_bytes).decode("ascii")
            logger.debug(
                "Aggressively resized image for LLM context: %d -> %d bytes",
                len(image_bytes),
                len(result_bytes),
            )
            return f"data:image/jpeg;base64,{encoded}"

    except Exception as e:
        logger.warning("Failed to resize image for LLM context: %s", e)
        return None


__all__ = ["resize_image_data_url_for_llm"]
