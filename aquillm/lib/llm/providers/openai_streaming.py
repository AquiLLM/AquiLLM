"""Streaming chat.completions consumption for OpenAI-compatible APIs."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from time import perf_counter
from typing import Any

import structlog

from ..types.response import LLMResponse
from .openai_tool_text import (
    decode_json_dict,
    extract_tool_call_from_text,
    is_textual_tool_call_only,
)
from .request_observability import (
    UsageCounts,
    authorized_tool_name,
    extract_usage,
    log_request_completed,
    new_correlation_id,
    safe_stage,
)
from .visibility import strip_thinking_blocks, visible_stream_content

logger = structlog.stdlib.get_logger(__name__)


async def consume_streaming_completion(
    *,
    stream: Any,
    stream_callback: Callable[..., Any],
    stream_message_uuid: str,
    raw_tools: list[dict] | None,
    model_name: str,
    correlation_id: str | None = None,
    stage: str = "general_answer",
    configured_max_tokens: int = 0,
    effective_max_tokens: int = 0,
    thinking_requested: bool = False,
    request_started_at: float | None = None,
    monotonic: Callable[[], float] = perf_counter,
) -> LLMResponse:
    request_started_at = (
        request_started_at if request_started_at is not None else monotonic()
    )
    correlation_id = correlation_id or new_correlation_id()
    stage = safe_stage(stage)
    text_parts: list[str] = []
    tool_call_parts: dict[int, dict[str, Any]] = {}
    finish_reason = "stop"
    input_usage = 0
    output_usage = 0
    reasoning_usage: int | None = None
    first_signal_at: float | None = None

    async for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if choices:
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                content_piece = getattr(delta, "content", None)
                reasoning_piece = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                delta_tool_calls = getattr(delta, "tool_calls", None) or []
                if first_signal_at is None and (
                    content_piece or reasoning_piece or delta_tool_calls
                ):
                    first_signal_at = monotonic()
                if content_piece:
                    piece = str(content_piece)
                    text_parts.append(piece)
                    visible_content = visible_stream_content(
                        "".join(text_parts),
                        raw_tools=raw_tools,
                        done=False,
                    )
                    if visible_content:
                        await stream_callback(
                            {
                                "message_uuid": stream_message_uuid,
                                "role": "assistant",
                                "content": visible_content,
                                "done": False,
                            }
                        )

                for tc in delta_tool_calls:
                    idx = int(getattr(tc, "index", 0) or 0)
                    entry = tool_call_parts.setdefault(
                        idx,
                        {"id": None, "name_parts": [], "arg_parts": []},
                    )
                    tc_id = getattr(tc, "id", None)
                    if tc_id:
                        entry["id"] = str(tc_id)
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        fn_name = getattr(fn, "name", None)
                        if fn_name:
                            entry["name_parts"].append(str(fn_name))
                        fn_args = getattr(fn, "arguments", None)
                        if fn_args:
                            entry["arg_parts"].append(str(fn_args))

            finish_reason_chunk = getattr(choice, "finish_reason", None)
            if finish_reason_chunk:
                finish_reason = str(finish_reason_chunk)

        usage = getattr(chunk, "usage", None)
        if usage is not None:
            usage_counts = extract_usage(usage)
            input_usage = usage_counts.prompt_tokens
            output_usage = usage_counts.completion_tokens
            reasoning_usage = usage_counts.reasoning_tokens

    text = strip_thinking_blocks("".join(text_parts)) or None
    tool_call_payload: dict | None = None
    if tool_call_parts:
        first_idx = sorted(tool_call_parts.keys())[0]
        first_tool_call = tool_call_parts[first_idx]
        tool_name = "".join(first_tool_call["name_parts"]).strip()
        tool_args = "".join(first_tool_call["arg_parts"])
        if tool_name:
            tool_call_payload = {
                "tool_call_id": first_tool_call["id"] or str(uuid.uuid4()),
                "tool_call_name": tool_name,
                "tool_call_input": decode_json_dict(tool_args),
            }
    elif text and raw_tools:
        tool_call_payload = extract_tool_call_from_text(text, raw_tools)
        if tool_call_payload and is_textual_tool_call_only(text):
            text = None

    if (
        tool_call_payload
        and tool_call_payload.get("tool_call_name") == "message_to_user"
    ):
        parsed_args = tool_call_payload.get("tool_call_input") or {}
        text = parsed_args.get("message") or text
        tool_call_payload = None

    visible_done_content = visible_stream_content(
        text or "",
        raw_tools=raw_tools,
        done=True,
        tool_call_payload=tool_call_payload,
    )
    if visible_done_content:
        await stream_callback(
            {
                "message_uuid": stream_message_uuid,
                "role": "assistant",
                "content": visible_done_content,
                "done": True,
                "stop_reason": finish_reason,
                "usage": input_usage + output_usage,
            }
        )

    completed_at = monotonic()
    usage_counts = UsageCounts(
        prompt_tokens=input_usage,
        completion_tokens=output_usage,
        reasoning_tokens=reasoning_usage,
    )
    log_request_completed(
        logger=logger,
        correlation_id=correlation_id,
        stage=stage,
        configured_max_tokens=configured_max_tokens,
        effective_max_tokens=effective_max_tokens,
        thinking_requested=thinking_requested,
        usage=usage_counts,
        finish_reason=finish_reason,
        request_started_at=request_started_at,
        first_signal_at=first_signal_at,
        completed_at=completed_at,
        tool_name=authorized_tool_name(tool_call_payload, raw_tools),
    )

    return LLMResponse(
        text=text,
        tool_call=tool_call_payload or {},
        stop_reason=finish_reason,
        input_usage=input_usage,
        output_usage=output_usage,
        reasoning_usage=reasoning_usage,
        model=model_name,
        message_uuid=stream_message_uuid,
    )


__all__ = ["consume_streaming_completion"]
