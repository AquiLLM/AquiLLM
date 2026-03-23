"""Build data URLs and read image bytes for document multimodal payloads."""
from __future__ import annotations

import base64
import logging
import mimetypes
from os import getenv
from typing import Any

from django.core.files.storage import default_storage

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        value = int((getenv(name) or str(default)).strip())
    except Exception:
        value = default
    return value if value > 0 else default


def _extract_image_bytes(doc: Any) -> tuple[bytes, str] | None:
    image_file = getattr(doc, "image_file", None)
    if not image_file:
        return None
    file_name = getattr(image_file, "name", "") or ""
    try:
        if file_name and default_storage.exists(file_name):
            with default_storage.open(file_name, "rb") as f:
                return f.read(), file_name
    except Exception as exc:
        logger.warning("_extract_image_bytes storage read failed for %r: %s", file_name, exc)

    if hasattr(image_file, "read"):
        position = None
        try:
            if hasattr(image_file, "tell"):
                position = image_file.tell()
            if hasattr(image_file, "seek"):
                image_file.seek(0)
            data = image_file.read()
            if isinstance(data, bytes) and data:
                return data, file_name
        except Exception as exc:
            logger.warning("_extract_image_bytes file-object read failed: %s", exc)
            return None
        finally:
            try:
                if position is not None and hasattr(image_file, "seek"):
                    image_file.seek(position)
            except Exception:
                pass
    return None


def _resize_image_to_fit(image_bytes: bytes, max_bytes: int, file_name: str = "") -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        import io as _io

        with Image.open(_io.BytesIO(image_bytes)) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            quality = 85
            max_dimension = 1600
            for _ in range(5):
                width, height = img.size
                if width > max_dimension or height > max_dimension:
                    if width > height:
                        new_width = max_dimension
                        new_height = int(height * (max_dimension / width))
                    else:
                        new_height = max_dimension
                        new_width = int(width * (max_dimension / height))
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                output = _io.BytesIO()
                img.save(output, format="JPEG", quality=quality, optimize=True)
                result = output.getvalue()
                if len(result) <= max_bytes:
                    return result

                max_dimension = int(max_dimension * 0.75)
                quality = max(40, quality - 10)
    except Exception as exc:
        logger.warning("Failed resizing image %r: %s", file_name, exc)
    return None


def _to_data_url(image_bytes: bytes, file_name: str = "") -> str | None:
    if not image_bytes:
        return None
    max_bytes = _env_int("APP_RAG_IMAGE_MAX_BYTES", 350_000)
    if len(image_bytes) > max_bytes:
        resized = _resize_image_to_fit(image_bytes, max_bytes, file_name)
        if resized is None:
            return None
        image_bytes = resized
        mime_type = "image/jpeg"
    else:
        mime_type = mimetypes.guess_type(file_name or "")[0] or "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def doc_image_data_url(doc: Any) -> str | None:
    extracted = _extract_image_bytes(doc)
    if not extracted:
        return None
    image_bytes, file_name = extracted
    return _to_data_url(image_bytes, file_name)


__all__ = ["doc_image_data_url"]
