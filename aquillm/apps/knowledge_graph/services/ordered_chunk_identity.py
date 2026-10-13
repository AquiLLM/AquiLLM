"""Exact ordered chunk identity and bounded identity text validation."""

from __future__ import annotations

from hashlib import sha256

from .representation_build_identity import _identity_key


def _text(value: object, label: str, *, maximum: int = 512) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(character in value for character in "\x00\r\n")
    ):
        raise ValueError(f"{label} must be a safe nonempty bounded string")
    return value


def ordered_chunk_signature(chunks, *, concrete_model_label: str = "") -> str:
    """Hash the exact ordered extraction inputs without retaining chunk text."""

    rows = tuple(chunks)
    payload = []
    previous_number = None
    seen_ids: set[int] = set()
    for chunk in rows:
        chunk_id = getattr(chunk, "pk", None)
        chunk_number = getattr(chunk, "chunk_number", None)
        content = getattr(chunk, "content", None)
        if (
            type(chunk_id) is not int
            or chunk_id <= 0
            or chunk_id in seen_ids
            or type(chunk_number) is not int
            or chunk_number < 0
            or (previous_number is not None and chunk_number <= previous_number)
            or type(content) is not str
        ):
            raise ValueError("ordered chunk snapshot is invalid")
        seen_ids.add(chunk_id)
        previous_number = chunk_number
        payload.append(
            {
                "chunk_id": chunk_id,
                "document_id": str(getattr(chunk, "doc_id", "")),
                "chunk_number": chunk_number,
                "start": getattr(chunk, "start_position", None),
                "end": getattr(chunk, "end_position", None),
                "modality": getattr(chunk, "modality", None),
                "content_hash": sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    return _identity_key(
        "kg-ordered-chunks-v1",
        {
            "concrete_model_label": _text(
                concrete_model_label or "unknown-document-model",
                "concrete document model",
                maximum=128,
            ),
            "chunks": tuple(payload),
        },
    )
