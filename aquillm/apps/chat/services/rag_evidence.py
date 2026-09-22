"""Evidence packet building for direct RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.chat.services.rag_config import evidence_token_budget, max_snippets_per_doc
from apps.chat.services.rag_legacy_selection import diversify_evidence_chunks
from apps.chat.services.rag_selection_types import EvidenceSelection
from lib.llm.providers.rag_citations import _chunk_citation_from_row

_CHARS_PER_TOKEN = 4
_PUBLIC_ROW_KEYS = frozenset(
    {
        "rank",
        "chunk_id",
        "doc_id",
        "chunk",
        "title",
        "citation",
        "text",
        "type",
        "image_url",
        "r",
        "i",
        "d",
        "c",
        "n",
        "ref",
        "x",
        "ty",
        "u",
    }
)


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
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _chunk_text(chunk: dict) -> str:
    """Extract text from either compact or full chunk format."""
    return chunk.get("text") or chunk.get("x") or ""


def _assemble_packet(
    selected: list[dict],
    *,
    query: str,
    search_scope: str,
    retrieval_status: str,
    diagnostic: str,
    total_tokens: int,
) -> EvidencePacket:
    citation_tokens: list[str] = []
    image_urls: list[str] = []
    for chunk in selected:
        token = _chunk_citation_from_row(chunk)
        if token and token not in citation_tokens:
            citation_tokens.append(token)
        url = chunk.get("image_url") or chunk.get("u")
        if (
            isinstance(url, str)
            and url.startswith("/aquillm/")
            and url not in image_urls
        ):
            image_urls.append(url)
    return EvidencePacket(
        chunks=selected,
        image_urls=image_urls,
        citation_tokens=citation_tokens,
        query=query,
        search_scope=search_scope,
        retrieval_status=retrieval_status if selected else "no_results",
        diagnostic_message=diagnostic,
        total_tokens=total_tokens,
    )


def build_selected_evidence_packet(
    selection: EvidenceSelection,
    *,
    query: str,
    search_scope: str,
) -> EvidencePacket:
    """Package already selected rows without another ordering or budget pass."""
    rows = [
        {key: value for key, value in candidate.row.items() if key in _PUBLIC_ROW_KEYS}
        for candidate in selection.candidates
    ]
    return _assemble_packet(
        rows,
        query=query,
        search_scope=search_scope,
        retrieval_status="results_found",
        diagnostic="" if rows else "No authorized evidence remains for this request.",
        total_tokens=sum(_estimate_tokens(_chunk_text(row)) for row in rows),
    )


def build_evidence_packet(
    raw_tool_result: dict[str, Any],
    *,
    query: str,
    search_scope: str,
    token_budget: int | None = None,
) -> EvidencePacket:
    """Normalise chunk search results into a token-budgeted evidence packet.

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
    capped = diversify_evidence_chunks(raw_chunks, per_doc_limit)

    # Skip passages that cannot fit so shorter evidence from later papers can
    # still be selected. Never send an oversized first passage past the budget.
    selected: list[dict] = []
    used_tokens = 0
    for chunk in capped:
        chunk_tokens = _estimate_tokens(_chunk_text(chunk))
        if used_tokens + chunk_tokens > budget:
            continue
        selected.append(chunk)
        used_tokens += chunk_tokens

    return _assemble_packet(
        selected,
        query=query,
        search_scope=search_scope,
        retrieval_status=retrieval_status,
        diagnostic=""
        if selected
        else "Retrieved passages could not fit within the evidence budget.",
        total_tokens=used_tokens,
    )


__all__ = [
    "EvidencePacket",
    "build_evidence_packet",
    "build_selected_evidence_packet",
    "diversify_evidence_chunks",
]
