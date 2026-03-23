"""Document-derived metadata for ingestion UI and chunking (no compat-layer imports)."""
from __future__ import annotations

from typing import Any


def document_modality(doc: Any) -> str:
    if hasattr(doc, "media_kind"):
        media_kind = (getattr(doc, "media_kind", "") or "").strip().lower()
        if media_kind in {"audio", "video"}:
            return media_kind
    if hasattr(doc, "image_file"):
        return "image"
    if hasattr(doc, "audio_file"):
        return "transcript"
    return "text"


def document_has_raw_media(doc: Any) -> bool:
    return bool(getattr(doc, "image_file", None) or getattr(doc, "media_file", None))


def document_provider_name(doc: Any) -> str:
    for field_name in ("ocr_provider", "transcribe_provider"):
        value = (getattr(doc, field_name, "") or "").strip()
        if value:
            return value
    return ""


def document_provider_model(doc: Any) -> str:
    for field_name in ("ocr_model", "transcribe_model"):
        value = (getattr(doc, field_name, "") or "").strip()
        if value:
            return value
    return ""


__all__ = [
    "document_modality",
    "document_has_raw_media",
    "document_provider_name",
    "document_provider_model",
]
