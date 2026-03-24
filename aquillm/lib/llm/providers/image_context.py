"""Image / markdown helpers for LLM message and tool-result context."""
from __future__ import annotations

import re
from json import dumps
from typing import Any

from ..types.conversation import Conversation
from ..types.messages import ToolMessage


def sanitize_data_urls_for_llm_text(text: str) -> str:
    return re.sub(
        r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+",
        "[image data url redacted for context budget]",
        text or "",
        flags=re.IGNORECASE,
    )


def serialize_tool_result_for_llm(result_dict: Any) -> str:
    """
    Build tool result text for LLM context while excluding private transport keys
    (e.g. _images blobs) and redacting inline base64 data URLs.
    """
    if isinstance(result_dict, dict):
        visible = {
            key: value
            for key, value in result_dict.items()
            if (not str(key).startswith("_")) or str(key) == "_image_instruction"
        }
        serialized = dumps(visible, ensure_ascii=False, default=str)
        return sanitize_data_urls_for_llm_text(serialized)
    return sanitize_data_urls_for_llm_text(str(result_dict))


def contains_markdown_image(text: str) -> bool:
    return bool(re.search(r"!\[[^\]]*\]\([^)]+\)", text or ""))


def looks_like_image_display_request(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    cues = (
        "show me",
        "display",
        "render",
        "show it",
        "display it",
        "in chat",
        "show image",
        "display image",
        "figure",
        "plot",
        "graph",
    )
    return any(cue in normalized for cue in cues)


def _result_row_image_url(value: dict) -> str | None:
    """Resolve image URL from verbose (`image_url`) or compact (`u` + `ty`) vector_search rows."""
    url = value.get("image_url")
    if isinstance(url, str):
        u = url.strip()
        if u:
            return u
    ty = value.get("type") or value.get("ty")
    compact = value.get("u")
    if isinstance(compact, str) and ty in ("image", "text_with_image"):
        u = compact.strip()
        if u:
            return u
    return None


def _result_row_image_caption(value: dict, key_fallback: str | None = None) -> str:
    for k in ("text", "x", "title", "n"):
        raw = value.get(k)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()[:80]
    if key_fallback and str(key_fallback).strip():
        return str(key_fallback).strip()[:80]
    return "Image"


def recent_tool_image_markdown(conversation: Conversation, max_images: int = 3) -> list[str]:
    lines: list[str] = []
    seen_urls: set[str] = set()
    tool_messages = [
        msg
        for msg in reversed(conversation.messages)
        if isinstance(msg, ToolMessage) and msg.for_whom == "assistant"
    ]
    for tool_msg in tool_messages[:4]:
        result_dict = tool_msg.result_dict if isinstance(tool_msg.result_dict, dict) else {}
        payload = result_dict.get("result")
        candidates: list[tuple[str, str]] = []

        if isinstance(payload, list):
            for value in payload:
                if not isinstance(value, dict):
                    continue
                url = _result_row_image_url(value)
                if url:
                    caption = _result_row_image_caption(value)
                    candidates.append((caption, url))

        if isinstance(payload, dict):
            direct_url = _result_row_image_url(payload)
            if direct_url:
                candidates.append((_result_row_image_caption(payload), direct_url))
            for key, value in payload.items():
                if isinstance(value, dict):
                    url = _result_row_image_url(value)
                    if url:
                        candidates.append((_result_row_image_caption(value, key_fallback=str(key)), url))

        for caption, url in candidates:
            if not url or url.startswith("data:image/"):
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            lines.append(f"![{caption}]({url})")
            if len(lines) >= max_images:
                return lines

    return lines


__all__ = [
    "contains_markdown_image",
    "looks_like_image_display_request",
    "recent_tool_image_markdown",
    "sanitize_data_urls_for_llm_text",
    "serialize_tool_result_for_llm",
]
