"""Controlled evidence-selection baselines for offline evaluation."""
from __future__ import annotations

from apps.chat.services.rag_evidence import _chunk_text, _estimate_tokens


def sequential_select(chunks: list[dict], token_budget: int) -> dict:
    """Select ranked chunks sequentially with production token/stop semantics."""
    selected: list[dict] = []
    total_tokens = 0
    for chunk in chunks:
        chunk_tokens = _estimate_tokens(_chunk_text(chunk))
        if total_tokens + chunk_tokens > token_budget and selected:
            break
        selected.append(chunk)
        total_tokens += chunk_tokens

    citation_tokens: list[str] = []
    image_urls: list[str] = []
    for chunk in selected:
        citation = chunk.get("citation") or chunk.get("ref")
        if citation and citation not in citation_tokens:
            citation_tokens.append(citation)

        image_url = chunk.get("image_url") or chunk.get("u")
        if (
            isinstance(image_url, str)
            and image_url.startswith("/aquillm/")
            and image_url not in image_urls
        ):
            image_urls.append(image_url)

    return {
        "chunks": selected,
        "image_urls": image_urls,
        "citation_tokens": citation_tokens,
        "total_tokens": total_tokens,
        "overrun_tokens": max(0, total_tokens - token_budget),
    }


__all__ = ["sequential_select"]
