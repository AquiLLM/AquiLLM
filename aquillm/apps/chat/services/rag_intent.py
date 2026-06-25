"""RAG intent classification for chat messages."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_DOCUMENT_TARGET_RE = re.compile(
    r"\b(documents?|docs?|papers?|files?|selected collections?|sources?)\b",
    flags=re.IGNORECASE,
)
_DOCUMENT_SEARCH_ACTION_RE = re.compile(
    r"\b(search|check|find|scan|read|retrieve|query|consult)\b|"
    r"\blook\s+(?:at|in|through|up)\b",
    flags=re.IGNORECASE,
)
_DOCUMENT_FIGURE_TARGET_RE = re.compile(
    r"\b(figures?|figs?\.?|images?|visuals?|plots?|graphs?|charts?|diagrams?)\b",
    flags=re.IGNORECASE,
)
_DOCUMENT_FIGURE_ACTION_RE = re.compile(
    r"\b(show|display|render|include|explain|find|get|pull|open)\b",
    flags=re.IGNORECASE,
)
_LOCAL_TOOL_ACTION_RE = re.compile(
    r"\b("
    r"sky\s+subtraction|subtract\s+the\s+sky|flat[-\s]?field(?:ing)?|"
    r"point\s+source(?:s)?|detect\s+source(?:s)?|fits|uploaded\s+files?|"
    r"use\s+(?:the\s+)?tool|run\s+(?:the\s+)?tool"
    r")\b",
    flags=re.IGNORECASE,
)
_RETRY_REQUEST_RE = re.compile(
    r"^\s*(?:try again|retry|please retry|run it again|do that again)\s*[.!?]*\s*$",
    flags=re.IGNORECASE,
)


def _collection_backed_document_question(text: str, collection_ids: list) -> bool:
    """True when the user asks a question about documents in the selected collections."""
    if not collection_ids:
        return False
    lowered = text.lower()
    doc_cues = ("paper", "document", "doc", "article", "source", "collection", "this", "these")
    question_cues = ("what", "how", "why", "explain", "summarize", "describe", "tell me", "?")
    return any(c in lowered for c in doc_cues) and any(c in lowered for c in question_cues)


@dataclass
class ChatIntent:
    """Structured classification of a chat message's retrieval and tool intent."""

    requires_rag: bool
    wants_figures: bool
    wants_whole_document: bool
    is_retry: bool
    requires_local_tools: bool
    reason: str


def classify_chat_message(
    text: str,
    *,
    selected_collection_ids: list,
    prior_tools: Optional[list] = None,
    prior_tool_choice=None,
) -> ChatIntent:
    """Classify a chat message to determine retrieval and tool intent.

    Returns a ``ChatIntent`` dataclass whose fields drive routing decisions in
    ``_configure_append_tools`` and (when enabled) the direct RAG pipeline.
    """
    text = text or ""

    # Retry check takes priority over everything else.
    if bool(_RETRY_REQUEST_RE.match(text)):
        return ChatIntent(
            requires_rag=bool(prior_tools),
            wants_figures=False,
            wants_whole_document=False,
            is_retry=True,
            requires_local_tools=False,
            reason="retry_request",
        )

    # Local-tool (e.g. FITS processing) takes priority over document RAG.
    if bool(_LOCAL_TOOL_ACTION_RE.search(text)):
        return ChatIntent(
            requires_rag=False,
            wants_figures=False,
            wants_whole_document=False,
            is_retry=False,
            requires_local_tools=True,
            reason="local_tool_request",
        )

    wants_figures = bool(
        _DOCUMENT_FIGURE_TARGET_RE.search(text) and _DOCUMENT_FIGURE_ACTION_RE.search(text)
    )
    explicit_search = bool(
        _DOCUMENT_TARGET_RE.search(text) and _DOCUMENT_SEARCH_ACTION_RE.search(text)
    )
    collection_backed = _collection_backed_document_question(text, selected_collection_ids)

    requires_rag = wants_figures or explicit_search or collection_backed

    if wants_figures:
        reason = "figure_request"
    elif explicit_search:
        reason = "explicit_search"
    elif collection_backed:
        reason = "collection_backed_question"
    else:
        reason = "no_retrieval_needed"

    return ChatIntent(
        requires_rag=requires_rag,
        wants_figures=wants_figures,
        wants_whole_document=False,
        is_retry=False,
        requires_local_tools=False,
        reason=reason,
    )


__all__ = ["ChatIntent", "classify_chat_message"]
