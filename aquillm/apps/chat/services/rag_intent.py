"""RAG intent classification for chat messages."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
_SELECTED_COLLECTION_CLARIFICATION_RE = re.compile(
    r"^\s*(?:(?:the|those)\s+)?(?:ones?|documents?|docs?|files?|papers?)\s+in\s+"
    r"(?:the\s+)?selected\s+collections?\s*[.!?]*\s*$",
    flags=re.IGNORECASE,
)
_SMALL_TALK_RE = re.compile(
    r"^\s*(?:"
    r"(?:hi|hello|hey)(?:\s+there)?(?:\s*[,!]\s*(?:how\s+are\s+you|how(?:'s|\s+is)\s+it\s+going))?"
    r"|thanks?(?:\s+you)?(?:\s+(?:so|very)\s+much)?"
    r"|ok(?:ay)?|sounds?\s+good|got\s+it|understood"
    r")\s*[.!?]*\s*$",
    flags=re.IGNORECASE,
)
_UI_MANAGEMENT_RE = re.compile(
    r"\b(?:open|show|change|select|deselect|manage|edit|rename|delete|create)\b"
    r"[^.?!]*\b(?:collection\s+settings|collection\s+picker|collections?)\b|"
    r"\b(?:upload|sign\s+in|log\s+in|account\s+settings)\b",
    flags=re.IGNORECASE,
)
_CHAT_HISTORY_RE = re.compile(
    r"\b(?:past|previous|prior|earlier|old(?:er)?|other|last(?:\s+time)?)\s+"
    r"(?:chats?|conversations?|threads?|discussions?|sessions?)\b|"
    r"\b(?:chat|conversation|thread|discussion)\s+history\b|"
    r"\bwhat\s+did\s+we\s+(?:discuss|talk\s+about|say|decide|cover)\b",
    flags=re.IGNORECASE,
)


def _collection_backed_document_question(text: str, collection_ids: list) -> bool:
    """Treat selected collections as evidence for substantive knowledge turns."""
    if not collection_ids:
        return False
    normalized = " ".join((text or "").split()).strip()
    if not normalized:
        return False
    if _SMALL_TALK_RE.fullmatch(normalized):
        return False
    if _UI_MANAGEMENT_RE.search(normalized) or _CHAT_HISTORY_RE.search(normalized):
        return False
    if _SELECTED_COLLECTION_CLARIFICATION_RE.match(normalized):
        return True
    return bool(re.search(r"[A-Za-z0-9]", normalized))


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
    prior_tools: list | None = None,
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
        _DOCUMENT_FIGURE_TARGET_RE.search(text)
        and _DOCUMENT_FIGURE_ACTION_RE.search(text)
    )
    explicit_search = bool(
        _DOCUMENT_TARGET_RE.search(text) and _DOCUMENT_SEARCH_ACTION_RE.search(text)
    )
    collection_backed = _collection_backed_document_question(
        text, selected_collection_ids
    )

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
