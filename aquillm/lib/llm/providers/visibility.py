"""Presentation policy: what assistant text may leave the server toward the chat UI."""
from __future__ import annotations

import re
from typing import Optional

from ..types.messages import AssistantMessage
from . import fallback_heuristics as fb

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>[\s\S]*?</think>", flags=re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think\b[^>]*>[\s\S]*$", flags=re.IGNORECASE)
_CHUNK_CITATION_RE = re.compile(r"\[doc:[^\]\s]+\s+chunk:\d+\]", flags=re.IGNORECASE)
_DOC_CITATION_RE = re.compile(r"\[doc:[^\]\s]+\]", flags=re.IGNORECASE)
_STATUS_OPENING_RE = re.compile(
    r"^(?:retrieving|searching|looking|reading|fetching|loading|processing|analyzing|gathering)\b",
    flags=re.IGNORECASE,
)
_STREAM_PROMISE_PREFIX_RE = re.compile(
    r"^\s*(?:i(?:'ll| will)|let me)\b",
    flags=re.IGNORECASE,
)

_DEFAULT_MIN_DISPLAY_WORDS = 20


def strip_thinking_blocks(text: Optional[str]) -> str:
    """Remove inline reasoning blocks from assistant-visible text."""
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    cleaned = _OPEN_THINK_RE.sub("", cleaned)
    return cleaned


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def looks_like_status_only(text: Optional[str]) -> bool:
    """
    Short, in-progress status lines (e.g. "Retrieving the paper...") — structural, not phrase lists.
    """
    visible = strip_thinking_blocks(text).strip()
    if not visible:
        return False
    words = _word_count(visible)
    if words > 18:
        return False
    if _STATUS_OPENING_RE.match(visible):
        return True
    if visible.endswith(("...", "…")) and words < 15:
        return True
    return False


def _looks_like_streaming_promise_prefix(text: Optional[str]) -> bool:
    """Block early 'I'll …' / 'Let me …' tokens until the answer is clearly underway."""
    visible = strip_thinking_blocks(text).strip()
    if not visible:
        return False
    if not _STREAM_PROMISE_PREFIX_RE.match(visible):
        return False
    return _word_count(visible) < 40


def is_interim_assistant_text(text: Optional[str]) -> bool:
    """True for model work-in-progress prose or raw tool transcripts."""
    visible = strip_thinking_blocks(text).strip()
    if not visible:
        return False
    if looks_like_status_only(visible):
        return True
    return fb.looks_like_deferred_tool_intent(visible) or fb.looks_like_raw_tool_transcript(visible)


def is_displayable_answer_text(
    text: Optional[str],
    *,
    min_words: int = _DEFAULT_MIN_DISPLAY_WORDS,
) -> bool:
    """
    True when assistant prose is substantial enough to show as a finished answer bubble.

    Tool-call rows, streaming placeholders, and post-tool stubs stay hidden until this passes.
    """
    visible = strip_thinking_blocks(text).strip()
    if not visible:
        return False
    if is_interim_assistant_text(visible):
        return False
    if _CHUNK_CITATION_RE.search(visible) or _DOC_CITATION_RE.search(visible):
        if _word_count(visible) >= 8:
            return True
    if _word_count(visible) >= min_words:
        return True
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", visible))
    if sentence_count >= 2 and _word_count(visible) >= 12:
        return True
    return _word_count(visible) >= 2


def sanitize_assistant_text(text: Optional[str], *, suppress_interim: bool = True) -> str:
    """Return text safe for a normal assistant response bubble."""
    visible = strip_thinking_blocks(text)
    if suppress_interim and not is_displayable_answer_text(visible):
        return ""
    return visible


def assistant_content_for_frontend(message: AssistantMessage) -> str:
    """Map a persisted assistant row to user-visible bubble content."""
    if message.tool_call_name:
        return ""
    return sanitize_assistant_text(message.content, suppress_interim=True)


def should_append_citation_sources(text: Optional[str]) -> bool:
    """Sources footer is only for display-ready answers, not status stubs."""
    return is_displayable_answer_text(text)


def clean_response_failure_text(*, after_tool_result: bool) -> str:
    """Safe fallback when the model only produced hidden/interim text."""
    if after_tool_result:
        return (
            "I found supporting context, but could not produce a clean final answer. "
            "Please retry and I will answer directly."
        )
    return (
        "I could not complete that response cleanly. "
        "Please retry or simplify the request."
    )


def visible_stream_content(
    text: Optional[str],
    *,
    raw_tools: Optional[list[dict]],
    done: bool,
    tool_call_payload: Optional[dict] = None,
) -> str:
    """
    Return content safe to send through the live stream channel.

    While streaming, forward any non-interim prose so the UI grows smoothly.
    On the final chunk, apply the same display rules used for persisted answers.
    """
    _ = raw_tools  # call-site compatibility
    visible = strip_thinking_blocks(text)
    if tool_call_payload:
        return ""
    if is_interim_assistant_text(visible):
        return ""
    if not done and _looks_like_streaming_promise_prefix(visible):
        return ""
    if is_displayable_answer_text(visible):
        return visible
    if not done and visible.strip():
        return visible
    if done and visible.strip() and _word_count(visible) >= 2:
        return visible
    return ""


__all__ = [
    "assistant_content_for_frontend",
    "clean_response_failure_text",
    "is_displayable_answer_text",
    "is_interim_assistant_text",
    "looks_like_status_only",
    "sanitize_assistant_text",
    "should_append_citation_sources",
    "strip_thinking_blocks",
    "visible_stream_content",
]
