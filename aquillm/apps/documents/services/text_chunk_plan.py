"""Pure planning for persisted document text chunks."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunkSpec:
    content: str
    start_position: int
    end_position: int
    chunk_number: int


def plan_text_chunks(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[TextChunkSpec]:
    """Return the historical fixed-width, overlapping chunk plan for ``text``."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    pitch = chunk_size - overlap
    return [
        TextChunkSpec(
            content=text[start : min(start + chunk_size, len(text))],
            start_position=start,
            end_position=min(start + chunk_size, len(text)),
            chunk_number=chunk_number,
        )
        for chunk_number, start in enumerate(range(0, len(text), pitch))
    ]


__all__ = ["TextChunkSpec", "plan_text_chunks"]
