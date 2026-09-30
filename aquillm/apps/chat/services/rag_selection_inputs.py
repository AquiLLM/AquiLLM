"""Immutable identity for the exact evidence offered to final relevance scoring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from apps.documents.services.chunk_rerank_results import (
    fingerprint_pair,
    fingerprint_text,
)


@dataclass(frozen=True, slots=True)
class FinalEvidenceInput:
    chunk_id: int
    source_fingerprint: str
    emitted_fingerprint: str
    pair: tuple[str, str]
    pair_fingerprint: str


def prepare_final_input(
    *,
    chunk_id: int,
    source_text: str,
    emitted_text: str,
    query: str,
    prepare_pair: Callable[[str, str], tuple[str, str] | None],
) -> FinalEvidenceInput | None:
    pair = prepare_pair(query, emitted_text)
    if pair is None or pair != (query, emitted_text):
        return None
    return FinalEvidenceInput(
        chunk_id,
        fingerprint_text(source_text),
        fingerprint_text(emitted_text),
        pair,
        fingerprint_pair(*pair),
    )


__all__ = ["FinalEvidenceInput", "prepare_final_input"]
