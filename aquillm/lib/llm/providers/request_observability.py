"""Privacy-safe usage, timing, and correlation helpers for LLM requests."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import isfinite
from typing import Any

_CORRELATION_ID: ContextVar[str | None] = ContextVar(
    "llm_observability_correlation_id",
    default=None,
)
_STAGE: ContextVar[str | None] = ContextVar(
    "llm_observability_stage",
    default=None,
)
_SAFE_STAGES = frozenset(
    {
        "tool_selection",
        "tool_retry",
        "direct_synthesis",
        "post_tool_synthesis",
        "general_answer",
        "citation_retry",
        "continuation",
    }
)
_SAFE_FINISH_REASONS = frozenset(
    {"stop", "length", "max_tokens", "tool_calls", "content_filter", "error"}
)
_SAFE_TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_SAFE_CORRELATION_ID = re.compile(
    r"(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


@dataclass(frozen=True, slots=True)
class UsageCounts:
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None


def _nonnegative_int(value: object, default: int = 0) -> int:
    if type(value) is int and value >= 0:
        return value
    return default


def extract_usage(usage: object | None) -> UsageCounts:
    """Extract only provider-reported counts; absence stays unavailable."""

    prompt_tokens = _nonnegative_int(getattr(usage, "prompt_tokens", None))
    completion_tokens = _nonnegative_int(getattr(usage, "completion_tokens", None))
    details = getattr(usage, "completion_tokens_details", None)
    raw_reasoning = getattr(details, "reasoning_tokens", None)
    reasoning_tokens = (
        raw_reasoning if type(raw_reasoning) is int and raw_reasoning >= 0 else None
    )
    return UsageCounts(prompt_tokens, completion_tokens, reasoning_tokens)


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def safe_correlation_id(value: object | None) -> str:
    candidate = str(value or "").strip().lower()
    if _SAFE_CORRELATION_ID.fullmatch(candidate):
        return candidate
    return new_correlation_id()


def current_correlation_id() -> str | None:
    return _CORRELATION_ID.get()


def current_stage() -> str | None:
    return _STAGE.get()


@contextmanager
def observability_scope(
    correlation_id: str,
    stage: str | None = None,
) -> Iterator[None]:
    correlation_token = _CORRELATION_ID.set(safe_correlation_id(correlation_id))
    stage_token = _STAGE.set(stage if stage in _SAFE_STAGES else None)
    try:
        yield
    finally:
        _STAGE.reset(stage_token)
        _CORRELATION_ID.reset(correlation_token)


def safe_stage(value: object | None, *, default: str = "general_answer") -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in _SAFE_STAGES else default


def authorized_tool_name(
    tool_call: Mapping[str, Any] | None,
    raw_tools: Sequence[Mapping[str, Any]] | None,
) -> str | None:
    if not tool_call or not raw_tools:
        return None
    candidate = str(tool_call.get("tool_call_name") or "").strip()
    if _SAFE_TOOL_NAME.fullmatch(candidate) is None:
        return None
    allowed: set[str] = set()
    for tool in raw_tools:
        name = tool.get("name")
        function = tool.get("function")
        if not name and isinstance(function, Mapping):
            name = function.get("name")
        if isinstance(name, str):
            allowed.add(name)
    return candidate if candidate in allowed else None


def _safe_elapsed_ms(start: float, end: float | None) -> float | str:
    if end is None:
        return "unavailable"
    elapsed = (end - start) * 1000.0
    if not isfinite(elapsed) or elapsed < 0:
        return "unavailable"
    return round(elapsed, 1)


def log_request_completed(
    *,
    logger: Any,
    correlation_id: str,
    stage: str,
    configured_max_tokens: int,
    effective_max_tokens: int,
    thinking_requested: bool,
    usage: UsageCounts,
    finish_reason: str,
    request_started_at: float,
    first_signal_at: float | None,
    completed_at: float,
    tool_name: str | None,
) -> None:
    """Emit a content-free provider completion event."""

    safe_finish_reason = str(finish_reason or "").strip().lower()
    if safe_finish_reason not in _SAFE_FINISH_REASONS:
        safe_finish_reason = "unknown"
    safe_tool = (
        tool_name
        if isinstance(tool_name, str) and _SAFE_TOOL_NAME.fullmatch(tool_name)
        else None
    )
    fields: dict[str, object] = {
        "correlation_id": safe_correlation_id(correlation_id),
        "stage": safe_stage(stage),
        "configured_max_tokens": _nonnegative_int(configured_max_tokens),
        "effective_max_tokens": _nonnegative_int(effective_max_tokens),
        "thinking_requested": bool(thinking_requested),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": (
            usage.reasoning_tokens
            if usage.reasoning_tokens is not None
            else "unavailable"
        ),
        "finish_reason": safe_finish_reason,
        "ttft_ms": _safe_elapsed_ms(request_started_at, first_signal_at),
        "duration_ms": _safe_elapsed_ms(request_started_at, completed_at),
    }
    if safe_tool:
        fields["tool_name"] = safe_tool
    logger.info("obs.llm.request_completed", **fields)


__all__ = [
    "UsageCounts",
    "authorized_tool_name",
    "current_correlation_id",
    "current_stage",
    "extract_usage",
    "log_request_completed",
    "new_correlation_id",
    "observability_scope",
    "safe_correlation_id",
    "safe_stage",
]
