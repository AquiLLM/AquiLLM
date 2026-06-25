"""Evidence packet building for direct RAG."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apps.chat.services.rag_config import evidence_token_budget, max_snippets_per_doc

_CHARS_PER_TOKEN = 4


@dataclass
class EvidencePacket:
    """Normalised, token-budgeted evidence structure ready for synthesis."""

    chunks: list[dict]
    image_urls: list[str]
    citation_tokens: list[str]
    query: str
    search_scope: str
    retrieval_status: str
    diagnostic_message: str
    total_tokens: int


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _chunk_text(chunk: dict) -> str:
    """Extract the textual content from a chunk dict (supports both compact and full formats)."""
    return chunk.get("text") or chunk.get("x") or ""


def _apply_per_doc_cap_and_diversify(
    chunks: list[dict],
    per_doc_limit: int,
) -> list[dict]:
    """Round-robin across documents so no single doc consumes all snippet slots.

    The strategy:
    1. Group chunks by ``doc_id`` preserving original ranking order within each group.
    2. Round-robin: take one chunk per doc per round until each doc hits its cap.

    This guarantees that when multiple docs are present, all get at least one
    snippet before any doc receives a second.
    """
    from collections import defaultdict

    doc_order: list[str] = []
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        doc_id = chunk.get("doc_id") or chunk.get("d", "")
        if doc_id not in doc_order:
            doc_order.append(doc_id)
        by_doc[doc_id].append(chunk)

    result: list[dict] = []
    doc_counts: dict[str, int] = {d: 0 for d in doc_order}
    doc_iters = {d: iter(by_doc[d]) for d in doc_order}
    exhausted: set[str] = set()

    while len(exhausted) < len(doc_order):
        for doc_id in doc_order:
            if doc_id in exhausted:
                continue
            if doc_counts[doc_id] >= per_doc_limit:
                exhausted.add(doc_id)
                continue
            try:
                chunk = next(doc_iters[doc_id])
                result.append(chunk)
                doc_counts[doc_id] += 1
            except StopIteration:
                exhausted.add(doc_id)

    return result


def build_evidence_packet(
    raw_tool_result: dict[str, Any],
    *,
    query: str,
    search_scope: str,
    token_budget: int | None = None,
) -> EvidencePacket:
    """Normalise a ``pack_chunk_search_results`` dict into a token-budgeted evidence packet.

    Enforces:
    - ``RAG_MAX_SNIPPETS_PER_DOC`` – no single document dominates snippet slots.
    - ``token_budget`` (defaults to ``RAG_EVIDENCE_TOKEN_BUDGET``) – total token cap.
    - Citation token extraction (``[doc:X chunk:Y]`` / ``ref`` field).
    - Image URL collection from ``image_url`` / ``u`` fields.
    - ``no_results`` → empty chunks + diagnostic message (no exception raised).
    """
    budget = token_budget if token_budget is not None else evidence_token_budget()
    per_doc_limit = max_snippets_per_doc()

    retrieval_status: str = raw_tool_result.get("retrieval_status", "results_found")
    raw_chunks: list[dict] = list(raw_tool_result.get("result") or [])

    if not raw_chunks or retrieval_status == "no_results":
        diagnostic = (
            raw_tool_result.get("retrieval_message")
            or f'Retrieval returned no relevant passages for "{query}".'
        )
        return EvidencePacket(
            chunks=[],
            image_urls=[],
            citation_tokens=[],
            query=query,
            search_scope=search_scope,
            retrieval_status="no_results",
            diagnostic_message=diagnostic,
            total_tokens=0,
        )

    # Apply per-doc cap with round-robin diversification.
    capped = _apply_per_doc_cap_and_diversify(raw_chunks, per_doc_limit)

    # Apply token budget: keep chunks in order until budget exhausted.
    selected: list[dict] = []
    used_tokens = 0
    for chunk in capped:
        chunk_tokens = _estimate_tokens(_chunk_text(chunk))
        if used_tokens + chunk_tokens > budget and selected:
            break
        selected.append(chunk)
        used_tokens += chunk_tokens

    # Extract citation tokens.
    citation_tokens: list[str] = []
    for chunk in selected:
        token = chunk.get("citation") or chunk.get("ref")
        if token and token not in citation_tokens:
            citation_tokens.append(token)

    # Collect image URLs.
    image_urls: list[str] = []
    for chunk in selected:
        url = chunk.get("image_url") or chunk.get("u")
        if url and isinstance(url, str) and url.startswith("/aquillm/") and url not in image_urls:
            image_urls.append(url)

    return EvidencePacket(
        chunks=selected,
        image_urls=image_urls,
        citation_tokens=citation_tokens,
        query=query,
        search_scope=search_scope,
        retrieval_status=retrieval_status,
        diagnostic_message="",
        total_tokens=used_tokens,
    )


__all__ = ["EvidencePacket", "build_evidence_packet"]
