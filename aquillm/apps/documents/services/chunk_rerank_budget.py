"""Deterministic token budgeting for reranker query/document pairs."""

from __future__ import annotations

from functools import lru_cache

from tiktoken import get_encoding


@lru_cache(maxsize=1)
def _encoding():
    return get_encoding("cl100k_base")


def _token_ids(text: str) -> list[int]:
    return _encoding().encode(text or "", disallowed_special=())


def _decode_prefix(tokens: list[int], limit: int) -> str:
    if limit <= 0:
        return ""
    return _encoding().decode(tokens[:limit])


def count_rerank_tokens(query: str, document: str) -> int:
    """Return the stable local token estimate used for pair budgeting."""

    return len(_token_ids(query)) + len(_token_ids(document))


def trim_rerank_pair(
    query: str,
    document: str,
    max_pair_tokens: int,
    reserve_tokens: int,
) -> tuple[str, str]:
    """Fit a pair within the model budget while preserving document evidence."""

    usable_tokens = max(2, int(max_pair_tokens) - max(0, int(reserve_tokens)))
    query_tokens = _token_ids(query)
    document_tokens = _token_ids(document)
    if len(query_tokens) + len(document_tokens) <= usable_tokens:
        return query, document

    if document_tokens:
        # Ordinary academic queries remain intact. Only an abnormally large query is
        # capped, and even then at least half of the pair remains evidence-bearing.
        document_floor = min(len(document_tokens), max(1, usable_tokens // 2))
        query_limit = min(len(query_tokens), usable_tokens - document_floor)
        document_limit = min(len(document_tokens), usable_tokens - query_limit)
    else:
        query_limit = usable_tokens
        document_limit = 0

    trimmed_query = _decode_prefix(query_tokens, query_limit)
    trimmed_document = _decode_prefix(document_tokens, document_limit)

    # Token decoding can normalize an incomplete Unicode boundary. Recheck and
    # remove document tail tokens until the authoritative local count fits.
    while (
        trimmed_document
        and count_rerank_tokens(trimmed_query, trimmed_document) > usable_tokens
    ):
        document_limit -= 1
        trimmed_document = _decode_prefix(document_tokens, document_limit)

    return trimmed_query, trimmed_document


__all__ = ["count_rerank_tokens", "trim_rerank_pair"]
