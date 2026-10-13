"""Retrieval query building for direct RAG."""

from __future__ import annotations

import re
from typing import Any


_RETRY_RE = re.compile(
    r"^\s*(?:try again|retry|please retry|run it again|do that again)\s*[.!?]*\s*$",
    flags=re.IGNORECASE,
)

def _is_retry(text: str) -> bool:
    return bool(_RETRY_RE.match(text))


def _last_vector_search_query(conversation: Any) -> str | None:
    """Return the ``search_string`` from the most recent vector_search tool call."""
    for msg in reversed(conversation.messages):
        if getattr(msg, "tool_call_name", None) == "vector_search" and isinstance(
            getattr(msg, "tool_call_input", None), dict
        ):
            query = msg.tool_call_input.get("search_string")
            if query:
                return str(query)
    return None


def build_retrieval_query(conversation: Any, latest_user_text: str) -> str:
    """Build the vector-search query string for the current turn.

    Resolution order:
    1. Retry phrases → reuse last ``vector_search`` query from history.
    Source references are resolved separately from validated displayed evidence.
    Retrieved-title ordering never establishes an antecedent.
    """
    text = (latest_user_text or "").strip()

    if _is_retry(text):
        prior = _last_vector_search_query(conversation)
        if prior:
            return prior
        return text

    return text


def build_retrieval_queries(
    conversation: Any,
    latest_user_text: str,
    *,
    max_queries: int,
) -> list[str]:
    """Keep all constraints together; optional additions use the shared ledger."""
    primary = build_retrieval_query(conversation, latest_user_text)
    if not primary:
        return []

    return [primary]


__all__ = ["build_retrieval_queries", "build_retrieval_query"]
