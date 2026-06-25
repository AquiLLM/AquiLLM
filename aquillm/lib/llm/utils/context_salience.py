"""Query relevance and lexical boosts for context retention ordering."""
from __future__ import annotations

import re
from typing import Any

from lib.llm.providers.openai_tokens import flatten_content_for_token_estimate


def _is_tool_evidence_msg(m: dict[str, Any]) -> bool:
    if str(m.get("role", "")).lower() != "user":
        return False
    content = m.get("content")
    if not isinstance(content, str):
        return False
    head = content.lstrip()[:160]
    first_line = head.split("\n", 1)[0]
    if first_line.startswith("Tool ") and "result:" in first_line:
        return True
    return first_line.startswith("Tool:")


def _word_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{3,}", s.lower()))


def _latest_primary_user_query(message_dicts: list[dict[str, Any]]) -> str:
    for m in reversed(message_dicts):
        if str(m.get("role", "")).lower() != "user":
            continue
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if _is_tool_evidence_msg(m):
            continue
        return content.strip()
    return ""


def lexical_overlap_score(query: str, text: str) -> float:
    q, t = _word_tokens(query), _word_tokens(text)
    if not q or not t:
        return 0.0
    inter = len(q & t)
    denom = max(1.0, (len(q) ** 0.5) * (len(t) ** 0.5))
    return float(inter) / denom


def entity_citation_boost(text: str) -> float:
    b = 0.0
    if re.search(r"\[\d+\]", text):
        b += 0.15
    if re.search(r"\b10\.\d{4,}/\S+", text) or "doi:" in text.lower():
        b += 0.2
    if re.search(r"\b(19|20)\d{2}\b", text):
        b += 0.05
    return min(b, 0.5)


def build_salience_scores(message_dicts: list[dict[str, Any]]) -> dict[int, float]:
    """
    Per-message salience vs latest primary user turn. Higher = keep longer under pressure.
    """
    query = _latest_primary_user_query(message_dicts)
    if not query:
        return {}
    n = len(message_dicts)
    out: dict[int, float] = {}
    for i, m in enumerate(message_dicts):
        flat = flatten_content_for_token_estimate(m.get("content", ""))
        overlap = lexical_overlap_score(query, flat)
        boost = entity_citation_boost(flat)
        recency = (i + 1) / max(n, 1) * 0.02
        out[i] = overlap + boost + recency
    return out


__all__ = [
    "build_salience_scores",
    "entity_citation_boost",
    "lexical_overlap_score",
]
