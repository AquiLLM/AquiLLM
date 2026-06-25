"""Retrieval query building for direct RAG."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from apps.chat.services.rag_config import query_rewrite_enabled

if TYPE_CHECKING:
    from lib.llm.types.conversation import Conversation

_RETRY_RE = re.compile(
    r"^\s*(?:try again|retry|please retry|run it again|do that again)\s*[.!?]*\s*$",
    flags=re.IGNORECASE,
)

_PRONOUN_CUES = frozenset({
    "it", "its", "itself",
    "they", "them", "their", "theirs",
    "this", "that", "these", "those",
})


def _is_retry(text: str) -> bool:
    return bool(_RETRY_RE.match(text))


def _has_pronoun_reference(text: str) -> bool:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return bool(words & _PRONOUN_CUES)


def _last_vector_search_query(conversation: Any) -> str | None:
    """Return the ``search_string`` from the most recent vector_search tool call."""
    for msg in reversed(conversation.messages):
        if (
            getattr(msg, "tool_call_name", None) == "vector_search"
            and isinstance(getattr(msg, "tool_call_input", None), dict)
        ):
            query = msg.tool_call_input.get("search_string")
            if query:
                return str(query)
    return None


def _last_retrieved_document_title(conversation: Any) -> str | None:
    """Return the first title from the most recent vector_search tool result."""
    for msg in reversed(conversation.messages):
        if (
            getattr(msg, "tool_name", None) == "vector_search"
            and isinstance(getattr(msg, "result_dict", None), dict)
        ):
            titles = msg.result_dict.get("retrieved_documents")
            if titles:
                return str(titles[0])
    return None


def _llm_rewrite_query(text: str, conversation: Any) -> str:
    """Placeholder for optional LLM-based query rewrite.

    In production this would call the LLM; in tests it is replaced via monkeypatch.
    When ``RAG_QUERY_REWRITE_ENABLED=1`` but no real LLM is wired here, the caller
    receives the original text unchanged (safe fallback).
    """
    return text


def build_retrieval_query(conversation: Any, latest_user_text: str) -> str:
    """Build the vector-search query string for the current turn.

    Resolution order:
    1. Retry phrases → reuse last ``vector_search`` query from history.
    2. Pronoun follow-ups → prepend the most recently retrieved document title.
    3. ``RAG_QUERY_REWRITE_ENABLED=1`` → call ``_llm_rewrite_query`` (mockable).
    4. Otherwise → return ``latest_user_text`` trimmed.
    """
    text = (latest_user_text or "").strip()

    if _is_retry(text):
        prior = _last_vector_search_query(conversation)
        if prior:
            return prior
        return text

    if _has_pronoun_reference(text):
        title = _last_retrieved_document_title(conversation)
        if title:
            return f"{title}: {text}"

    if query_rewrite_enabled():
        return _llm_rewrite_query(text, conversation)

    return text


__all__ = ["build_retrieval_query"]
