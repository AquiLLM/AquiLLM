"""OpenAI-compatible context overflow and timeout retry argument adjustment."""
from __future__ import annotations

import re
from os import getenv
from typing import Optional

from .openai_tokens import trim_messages_for_overflow


def context_overflow_search_text(exc: BaseException) -> str:
    """Collect all text the API might put the overflow message in (str vs nested JSON body)."""
    parts: list[str] = []
    s = str(exc)
    if s:
        parts.append(s)
    msg = getattr(exc, "message", None)
    if isinstance(msg, str) and msg.strip() and msg not in s:
        parts.append(msg)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err_obj = body.get("error")
        if isinstance(err_obj, dict):
            inner = err_obj.get("message")
            if isinstance(inner, str) and inner.strip():
                parts.append(inner)
        top = body.get("message")
        if isinstance(top, str) and top.strip():
            parts.append(top)
    return "\n".join(parts)


def strip_images_from_messages(arguments: dict) -> bool:
    """Remove image content from messages to recover from context overflow."""
    messages = arguments.get("messages")
    if not isinstance(messages, list):
        return False

    stripped = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type in {"image_url", "image", "input_image"}:
                        stripped = True
                        new_content.append(
                            {"type": "text", "text": "[Image removed due to context limit]"}
                        )
                    else:
                        new_content.append(part)
                else:
                    new_content.append(part)
            if stripped:
                text_parts = [
                    p.get("text", "")
                    for p in new_content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                if len(text_parts) == len(new_content):
                    msg["content"] = "\n".join(text_parts)
                else:
                    msg["content"] = new_content

    return stripped


def retry_args_for_context_overflow(arguments: dict, exc: Exception) -> Optional[dict]:
    """Parse context overflow error and adjust arguments for retry."""
    message = context_overflow_search_text(exc)
    match = re.search(
        r"passed\s+(\d+)\s+input tokens.*maximum input length of\s+(\d+)\s+tokens",
        message,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    passed_input_tokens = int(match.group(1))
    max_input_tokens = int(match.group(2))
    overflow = passed_input_tokens - max_input_tokens
    if overflow <= 0:
        return None
    current_max_tokens = int(arguments.get("max_tokens", 0))
    has_tools = bool(arguments.get("tools"))
    try:
        if has_tools:
            min_completion_tokens = max(64, int(getenv("LLM_TOOL_MIN_COMPLETION_TOKENS", "128")))
            hard_floor_tokens = 64
        else:
            min_completion_tokens = max(128, int(getenv("LLM_MIN_COMPLETION_TOKENS", "384")))
            hard_floor_tokens = 192
    except Exception:
        min_completion_tokens = 128 if has_tools else 384
        hard_floor_tokens = 64 if has_tools else 192

    retry_args = dict(arguments)
    if isinstance(arguments.get("messages"), list):
        retry_args["messages"] = [
            dict(msg) if isinstance(msg, dict) else msg for msg in arguments["messages"]
        ]

    changed = False

    if strip_images_from_messages(retry_args):
        changed = True

    if current_max_tokens > min_completion_tokens:
        if overflow <= 4:
            safety_margin = 8
        else:
            safety_margin = max(32, min(192, overflow * 4))
        reduced_max_tokens = max(
            min_completion_tokens, current_max_tokens - overflow - safety_margin
        )
        if reduced_max_tokens >= current_max_tokens:
            reduced_max_tokens = max(min_completion_tokens, current_max_tokens - 1)
        if reduced_max_tokens != current_max_tokens:
            retry_args["max_tokens"] = reduced_max_tokens
            changed = True

    if not changed and current_max_tokens > hard_floor_tokens:
        emergency_margin = max(16, min(128, overflow * 4))
        emergency_reduced_max_tokens = max(
            hard_floor_tokens,
            current_max_tokens - overflow - emergency_margin,
        )
        if emergency_reduced_max_tokens >= current_max_tokens:
            emergency_reduced_max_tokens = max(
                hard_floor_tokens,
                current_max_tokens - 1,
            )
        if emergency_reduced_max_tokens < current_max_tokens:
            retry_args["max_tokens"] = emergency_reduced_max_tokens
            changed = True

    should_trim_context = overflow > 0
    if should_trim_context and trim_messages_for_overflow(retry_args, max(overflow, 1)):
        changed = True

    if not changed and strip_images_from_messages(retry_args):
        changed = True

    return retry_args if changed else None


def retry_args_for_timeout(arguments: dict, attempt: int) -> Optional[dict]:
    current_max_tokens = int(arguments.get("max_tokens", 0))
    if current_max_tokens <= 0:
        return None
    has_tools = bool(arguments.get("tools"))
    try:
        min_completion_tokens = max(
            64 if has_tools else 128,
            int(getenv("LLM_TOOL_MIN_COMPLETION_TOKENS", "128"))
            if has_tools
            else int(getenv("LLM_MIN_COMPLETION_TOKENS", "256")),
        )
    except Exception:
        min_completion_tokens = 128 if has_tools else 256

    retry_args = dict(arguments)
    if isinstance(arguments.get("messages"), list):
        retry_args["messages"] = [
            dict(msg) if isinstance(msg, dict) else msg for msg in arguments["messages"]
        ]

    changed = False
    reduction_ratio = min(0.6, 0.2 + (0.1 * attempt))
    reduction_tokens = max(64, int(current_max_tokens * reduction_ratio))
    reduced_max_tokens = max(min_completion_tokens, current_max_tokens - reduction_tokens)
    if reduced_max_tokens < current_max_tokens:
        retry_args["max_tokens"] = reduced_max_tokens
        changed = True

    if attempt >= 2 and trim_messages_for_overflow(
        retry_args,
        256 * (attempt - 1),
    ):
        changed = True

    return retry_args if changed else None


__all__ = [
    "context_overflow_search_text",
    "retry_args_for_context_overflow",
    "retry_args_for_timeout",
    "strip_images_from_messages",
]
