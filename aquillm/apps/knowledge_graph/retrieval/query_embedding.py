"""Transient, signature-locked embedding for unresolved query spans."""

from __future__ import annotations

from math import isfinite
from time import monotonic


def _load_embedding_api():
    from aquillm.utils import (
        get_strict_index_embeddings,
        strict_index_embedding_signature,
    )

    return strict_index_embedding_signature, get_strict_index_embeddings


def embed_unresolved_query_span(
    *, text: str, expected_signature: str, deadline: float
) -> tuple[float, ...]:
    if (
        type(text) is not str
        or not text
        or len(text) > 8_192
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise ValueError("query span text must be bounded canonical text")
    if (
        type(expected_signature) is not str
        or not expected_signature
        or len(expected_signature) > 512
        or "dims=1024" not in expected_signature.split(":")
    ):
        raise ValueError("expected embedding signature is invalid")
    if type(deadline) is not float or not isfinite(deadline):
        raise ValueError("deadline must be an exact finite float")
    if monotonic() >= deadline:
        raise TimeoutError("query embedding deadline expired")
    signature_loader, embedding_loader = _load_embedding_api()
    actual_signature = signature_loader()
    if actual_signature != expected_signature:
        raise RuntimeError("query embedding signature mismatch")
    rows, returned_signature = embedding_loader(
        [text], expected_model_signature=expected_signature
    )
    if monotonic() >= deadline:
        raise TimeoutError("query embedding deadline expired")
    if returned_signature != expected_signature or len(rows) != 1:
        raise RuntimeError("query embedding response provenance mismatch")
    index, vector = rows[0]
    if index != 0 or not isinstance(vector, (list, tuple)) or len(vector) != 1024:
        raise RuntimeError("query embedding response dimension mismatch")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        for value in vector
    ):
        raise RuntimeError("query embedding response contains invalid values")
    return tuple(float(value) for value in vector)


__all__ = ["embed_unresolved_query_span"]
