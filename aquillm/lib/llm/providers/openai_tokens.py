"""OpenAI-compatible prompt token estimation and preflight context trimming."""
from __future__ import annotations

from os import getenv
from typing import Any


def env_int(name: str, default: int) -> int:
    try:
        return int((getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float((getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return default


def context_reserve_tokens(context_limit: int) -> tuple[int, int]:
    mode = str(getenv("OPENAI_CONTEXT_RESERVE_MODE", "ratio") or "ratio").strip().lower()
    if mode == "fixed":
        guard_tokens = max(64, env_int("OPENAI_CONTEXT_GUARD_TOKENS", 256))
        estimator_pad_tokens = max(0, env_int("OPENAI_ESTIMATOR_PAD_TOKENS", 256))
        return guard_tokens, estimator_pad_tokens

    guard_ratio = min(max(env_float("OPENAI_CONTEXT_GUARD_RATIO", 0.015), 0.0), 0.5)
    pad_ratio = min(max(env_float("OPENAI_ESTIMATOR_PAD_RATIO", 0.0075), 0.0), 0.5)

    guard_min = max(64, env_int("OPENAI_CONTEXT_GUARD_MIN_TOKENS", 96))
    guard_max = max(guard_min, env_int("OPENAI_CONTEXT_GUARD_MAX_TOKENS", 4096))
    pad_min = max(0, env_int("OPENAI_ESTIMATOR_PAD_MIN_TOKENS", 64))
    pad_max = max(pad_min, env_int("OPENAI_ESTIMATOR_PAD_MAX_TOKENS", 2048))

    guard_tokens = int(context_limit * guard_ratio)
    estimator_pad_tokens = int(context_limit * pad_ratio)
    guard_tokens = min(guard_max, max(guard_min, guard_tokens))
    estimator_pad_tokens = min(pad_max, max(pad_min, estimator_pad_tokens))
    return guard_tokens, estimator_pad_tokens


def trim_messages_for_overflow(arguments: dict, overflow_tokens: int) -> bool:
    messages = arguments.get("messages")
    if not isinstance(messages, list) or len(messages) <= 1:
        return False

    if len(messages) >= 3:
        del messages[1]
        return True

    candidate_indices = [
        i
        for i in range(1, len(messages) - 1)
        if isinstance(messages[i], dict) and isinstance(messages[i].get("content"), str)
    ]
    if not candidate_indices:
        candidate_indices = [
            i
            for i in range(1, len(messages))
            if isinstance(messages[i], dict) and isinstance(messages[i].get("content"), str)
        ]
    if not candidate_indices:
        return False

    idx = candidate_indices[0]
    content = str(messages[idx].get("content", ""))
    if not content:
        return False

    trim_chars = max(128, overflow_tokens * 12)
    if len(content) <= trim_chars:
        messages[idx]["content"] = "[Earlier context trimmed due to token limit.]"
    else:
        messages[idx]["content"] = content[trim_chars:]
    return True


def flatten_content_for_token_estimate(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text_val = content.get("text")
        if isinstance(text_val, str):
            return text_val
        part_type = str(content.get("type", "")).lower()
        if part_type in {"image_url", "input_image", "image"}:
            image_val = content.get("image_url")
            if isinstance(image_val, dict):
                url_val = image_val.get("url")
                if isinstance(url_val, str):
                    return url_val
            if isinstance(image_val, str):
                return image_val
            direct_url = content.get("url")
            if isinstance(direct_url, str):
                return direct_url
        content_val = content.get("content")
        if isinstance(content_val, str):
            return content_val
        return str(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                part_type = str(item.get("type", "")).lower()
                if part_type in {"image_url", "input_image", "image"}:
                    image_val = item.get("image_url")
                    if isinstance(image_val, dict):
                        url_val = image_val.get("url")
                        if isinstance(url_val, str):
                            parts.append(url_val)
                            continue
                    if isinstance(image_val, str):
                        parts.append(image_val)
                        continue
                    direct_url = item.get("url")
                    if isinstance(direct_url, str):
                        parts.append(direct_url)
                        continue
                text_val = item.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
                    continue
                content_val = item.get("content")
                if isinstance(content_val, str):
                    parts.append(content_val)
        return "\n".join(parts)
    return str(content)


def estimate_prompt_tokens(messages: list[dict], encoder) -> int:
    total = 12
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", ""))
        content = flatten_content_for_token_estimate(msg.get("content", ""))
        total += 6 + len(encoder.encode(role)) + len(encoder.encode(content))
    return total


def preflight_trim_for_context(
    cls: type,
    arguments: dict,
    context_limit: int,
    extra_prompt_slack: int = 0,
) -> None:
    messages = arguments.get("messages")
    if not isinstance(messages, list) or len(messages) <= 1 or context_limit <= 0:
        return

    guard_tokens, estimator_pad_tokens = context_reserve_tokens(context_limit)
    slack = max(0, int(extra_prompt_slack))

    has_tools = bool(arguments.get("tools"))
    min_completion_tokens = 128 if has_tools else 256
    current_max_tokens = int(arguments.get("max_tokens", 0))
    if current_max_tokens <= 0:
        return

    prompt_budget = (
        context_limit - current_max_tokens - guard_tokens - estimator_pad_tokens - slack
    )
    if prompt_budget < 256:
        reduced_completion_tokens = max(
            min_completion_tokens,
            context_limit - guard_tokens - estimator_pad_tokens - slack - 256,
        )
        if 0 < reduced_completion_tokens < current_max_tokens:
            arguments["max_tokens"] = reduced_completion_tokens
            current_max_tokens = reduced_completion_tokens
            prompt_budget = (
                context_limit
                - current_max_tokens
                - guard_tokens
                - estimator_pad_tokens
                - slack
            )

    if prompt_budget <= 0:
        return

    prompt_tokens = cls._estimate_prompt_tokens(messages)
    trim_loops = 0
    while prompt_tokens > prompt_budget and trim_loops < 32:
        overflow_estimate = max(1, prompt_tokens - prompt_budget)
        if not cls._trim_messages_for_overflow(arguments, overflow_estimate):
            break
        messages = arguments.get("messages")
        if not isinstance(messages, list):
            break
        prompt_tokens = cls._estimate_prompt_tokens(messages)
        trim_loops += 1

    if prompt_tokens > prompt_budget:
        available_completion_tokens = (
            context_limit - prompt_tokens - guard_tokens - estimator_pad_tokens - slack
        )
        if min_completion_tokens <= available_completion_tokens < current_max_tokens:
            arguments["max_tokens"] = available_completion_tokens


__all__ = [
    "context_reserve_tokens",
    "env_float",
    "env_int",
    "estimate_prompt_tokens",
    "flatten_content_for_token_estimate",
    "preflight_trim_for_context",
    "trim_messages_for_overflow",
]
