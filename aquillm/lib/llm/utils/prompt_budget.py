"""Provider-agnostic preflight history trimming using OpenAI-shaped token estimates."""
from __future__ import annotations

import structlog
from os import getenv
from typing import Any

from tiktoken import encoding_for_model

from lib.llm.providers.openai_tokens import (
    estimate_prompt_tokens,
    preflight_trim_for_context,
    trim_messages_for_overflow,
)

logger = structlog.stdlib.get_logger(__name__)
_ENC = encoding_for_model("gpt-4o")


def _settings_int(name: str, default: int) -> int:
    try:
        from django.conf import settings

        if hasattr(settings, name):
            return int(getattr(settings, name))
    except Exception:
        pass
    try:
        return int((getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return default


def _settings_bool(name: str, default: bool = False) -> bool:
    try:
        from django.conf import settings

        if hasattr(settings, name):
            return bool(getattr(settings, name))
    except Exception:
        pass
    v = (getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def token_efficiency_enabled() -> bool:
    return _settings_bool("TOKEN_EFFICIENCY_ENABLED", False)


def context_packer_enabled() -> bool:
    return _settings_bool("CONTEXT_PACKER_ENABLED", False)


def prompt_budget_context_limit() -> int:
    v = _settings_int("PROMPT_BUDGET_CONTEXT_LIMIT", 0)
    if v > 0:
        return v
    for key in ("OPENAI_CONTEXT_LIMIT", "VLLM_MAX_MODEL_LEN"):
        raw = (getenv(key, "") or "").strip()
        if raw:
            try:
                return int(raw)
            except Exception:
                continue
    return 0


def prompt_budget_slack_tokens() -> int:
    return _settings_int("PROMPT_BUDGET_SLACK_TOKENS", 384)


def prompt_budget_max_tokens_cap() -> int:
    return _settings_int("PROMPT_BUDGET_MAX_TOKENS_CAP", 8192)


def maybe_pack_message_dicts_for_context(
    system_text: str,
    message_dicts: list[dict[str, Any]],
    *,
    context_limit: int,
    max_tokens: int,
) -> tuple[bool, int]:
    """
    Run salience-aware packing when CONTEXT_PACKER_ENABLED. Mutates message_dicts in place.
    Returns (changed?, effective_max_tokens).
    """
    if context_limit <= 0 or not context_packer_enabled():
        return (False, max_tokens)
    try:
        from lib.llm.utils.context_packer import load_context_packer_config, pack_messages_for_budget

        cfg = load_context_packer_config()
        mt = min(max(int(max_tokens), 1), prompt_budget_max_tokens_cap())
        before_messages = list(message_dicts)
        out = pack_messages_for_budget(
            system_text,
            message_dicts,
            context_limit,
            mt,
            cfg,
            slack=prompt_budget_slack_tokens(),
        )
        packed = out["messages"]
        new_max = int(out["max_tokens"])
        message_dicts[:] = packed
        changed = packed != before_messages or new_max != max_tokens
        return (changed, new_max)
    except Exception as exc:
        logger.warning("obs.llm.prompt_budget_error", error_type=type(exc).__name__)
        return (False, max_tokens)


def apply_preflight_trim_to_message_dicts(
    system_text: str,
    message_dicts: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[bool, int]:
    """
    Trim `message_dicts` in place (roles preserved) when over budget.
    Returns (trimmed?, effective_max_tokens).
    """
    limit = prompt_budget_context_limit()
    if limit <= 0:
        return (False, max_tokens)
    te = token_efficiency_enabled()
    if not te and not context_packer_enabled():
        return (False, max_tokens)

    orig_max = max(int(max_tokens), 1)
    mt = min(orig_max, prompt_budget_max_tokens_cap())
    changed_pack, mt = maybe_pack_message_dicts_for_context(
        system_text, message_dicts, context_limit=limit, max_tokens=mt
    )

    if not te:
        if changed_pack:
            tail_tok = estimate_prompt_tokens(
                [{"role": "system", "content": system_text}] + message_dicts,
                _ENC,
            )
            logger.info("obs.llm.prompt_budget_pack", estimated_input_tokens=tail_tok, max_tokens=mt)
        return (changed_pack, mt)

    class _Estimator:
        @staticmethod
        def _trim_messages_for_overflow(arguments: dict, overflow_tokens: int) -> bool:
            return trim_messages_for_overflow(arguments, overflow_tokens)

        @staticmethod
        def _estimate_prompt_tokens(messages: list[dict]) -> int:
            return estimate_prompt_tokens(messages, _ENC)

    arguments: dict[str, Any] = {
        "messages": [{"role": "system", "content": system_text}] + message_dicts,
        "max_tokens": mt,
    }
    before = estimate_prompt_tokens(arguments["messages"], _ENC)
    preflight_trim_for_context(_Estimator, arguments, limit, prompt_budget_slack_tokens())
    new_messages = arguments.get("messages")
    new_max = int(arguments.get("max_tokens", mt))
    if not isinstance(new_messages, list) or len(new_messages) < 2:
        return (False, max_tokens)
    trimmed_tail = new_messages[1:]
    after = estimate_prompt_tokens(new_messages, _ENC)
    changed = (
        changed_pack
        or after < before
        or trimmed_tail != message_dicts
        or new_max != orig_max
    )
    message_dicts[:] = trimmed_tail
    if changed:
        logger.info("obs.llm.prompt_budget_preflight", input_tokens_before=before, input_tokens_after=after, max_tokens=new_max)
    return (changed, new_max)


def sync_trimmed_dicts_into_pydantic_messages(
    messages_pydantic: list[Any] | None,
    trimmed_dicts: list[dict[str, Any]],
) -> None:
    """Copy string `content` from trimmed dicts onto parallel pydantic messages (best-effort)."""
    if not messages_pydantic:
        return
    for i, p in enumerate(messages_pydantic):
        if i >= len(trimmed_dicts):
            break
        d = trimmed_dicts[i]
        content = d.get("content")
        if not isinstance(content, str):
            continue
        cur = getattr(p, "content", None)
        if isinstance(cur, str):
            p.content = content


__all__ = [
    "apply_preflight_trim_to_message_dicts",
    "context_packer_enabled",
    "maybe_pack_message_dicts_for_context",
    "prompt_budget_context_limit",
    "prompt_budget_max_tokens_cap",
    "prompt_budget_slack_tokens",
    "sync_trimmed_dicts_into_pydantic_messages",
    "token_efficiency_enabled",
]
