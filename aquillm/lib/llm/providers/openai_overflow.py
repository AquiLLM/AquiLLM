"""OpenAI-compatible context overflow and timeout retry argument adjustment."""
from __future__ import annotations

import re
from os import getenv
from typing import Optional

from .openai_tokens import trim_messages_for_overflow


def _token_int(raw: str) -> int:
    """Parse token counts that may include separators like commas/underscores."""
    return int(re.sub(r"[,_\s]", "", raw))


def _extract_overflow_tokens(message: str) -> Optional[int]:
    """Extract overflow amount from common OpenAI-compatible context error templates."""
    if not message:
        return None

    patterns = (
        (
            r"passed\s+([\d,\s_]+)\s+input tokens.*maximum input length of\s+([\d,\s_]+)\s+tokens",
            lambda m: _token_int(m.group(1)) - _token_int(m.group(2)),
        ),
        (
            r"maximum context length is\s+([\d,\s_]+)\s+tokens.*requested\s+([\d,\s_]+)\s+tokens",
            lambda m: _token_int(m.group(2)) - _token_int(m.group(1)),
        ),
        (
            r"requested\s+([\d,\s_]+)\s+tokens.*maximum context length is\s+([\d,\s_]+)\s+tokens",
            lambda m: _token_int(m.group(1)) - _token_int(m.group(2)),
        ),
    )
    for pattern, overflow_fn in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        overflow = overflow_fn(match)
        if overflow > 0:
            return overflow
    return None


def context_overflow_search_text(exc: BaseException) -> str:
    """Collect all text the API might put the overflow message in (str vs nested JSON body)."""
    parts: list[str] = []
    seen_exc_ids: set[int] = set()
    queue: list[BaseException] = [exc]

    while queue:
        current = queue.pop(0)
        if id(current) in seen_exc_ids:
            continue
        seen_exc_ids.add(id(current))

        s = str(current)
        if isinstance(s, str) and s.strip():
            parts.append(s)

        msg = getattr(current, "message", None)
        if isinstance(msg, str) and msg.strip():
            parts.append(msg)

        body = getattr(current, "body", None)
        if isinstance(body, dict):
            err_obj = body.get("error")
            if isinstance(err_obj, dict):
                inner = err_obj.get("message")
                if isinstance(inner, str) and inner.strip():
                    parts.append(inner)
            top = body.get("message")
            if isinstance(top, str) and top.strip():
                parts.append(top)

        for arg in getattr(current, "args", ()):
            if isinstance(arg, str) and arg.strip():
                parts.append(arg)
            elif isinstance(arg, dict):
                nested_err = arg.get("error")
                if isinstance(nested_err, dict):
                    inner = nested_err.get("message")
                    if isinstance(inner, str) and inner.strip():
                        parts.append(inner)
                top = arg.get("message")
                if isinstance(top, str) and top.strip():
                    parts.append(top)

        nested = current.__cause__ or current.__context__
        if isinstance(nested, BaseException):
            queue.append(nested)
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
    overflow = _extract_overflow_tokens(message)
    if overflow is None:
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
