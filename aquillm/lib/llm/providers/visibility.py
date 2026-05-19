"""User-visible assistant text policy for streamed and persisted responses."""
from __future__ import annotations

import re
from typing import Optional

from . import fallback_heuristics as fb

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>[\s\S]*?</think>", flags=re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think\b[^>]*>[\s\S]*$", flags=re.IGNORECASE)


def strip_thinking_blocks(text: Optional[str]) -> str:
    """Remove inline reasoning blocks from assistant-visible text."""
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    cleaned = _OPEN_THINK_RE.sub("", cleaned)
    return cleaned


def is_interim_assistant_text(text: Optional[str]) -> bool:
    """True for model work-in-progress prose or raw tool transcripts."""
    visible = strip_thinking_blocks(text).strip()
    if not visible:
        return False
    return fb.looks_like_deferred_tool_intent(visible) or fb.looks_like_raw_tool_transcript(visible)


def sanitize_assistant_text(text: Optional[str], *, suppress_interim: bool = True) -> str:
    """Return text safe for a normal assistant response bubble."""
    visible = strip_thinking_blocks(text).strip()
    if suppress_interim and is_interim_assistant_text(visible):
        return ""
    return visible


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
    """Return content safe to send through the live stream channel."""
    visible = strip_thinking_blocks(text)
    if tool_call_payload:
        return ""
    if is_interim_assistant_text(visible):
        return ""
    if raw_tools:
        return visible if done else ""
    return visible


__all__ = [
    "clean_response_failure_text",
    "is_interim_assistant_text",
    "sanitize_assistant_text",
    "strip_thinking_blocks",
    "visible_stream_content",
]
