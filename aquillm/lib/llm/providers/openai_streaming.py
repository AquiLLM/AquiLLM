"""Streaming chat.completions consumption for OpenAI-compatible APIs."""
from __future__ import annotations

import re
import uuid
from typing import Any, Callable, Optional

from ..types.response import LLMResponse

from .openai_tool_text import decode_json_dict, extract_tool_call_from_text


async def consume_streaming_completion(
    *,
    stream: Any,
    stream_callback: Callable[..., Any],
    stream_message_uuid: str,
    raw_tools: Optional[list[dict]],
    model_name: str,
) -> LLMResponse:
    text_parts: list[str] = []
    tool_call_parts: dict[int, dict[str, Any]] = {}
    finish_reason = "stop"
    input_usage = 0
    output_usage = 0

    async for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if choices:
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                content_piece = getattr(delta, "content", None)
                if content_piece:
                    piece = str(content_piece)
                    text_parts.append(piece)
                    await stream_callback(
                        {
                            "message_uuid": stream_message_uuid,
                            "role": "assistant",
                            "content": "".join(text_parts),
                            "done": False,
                        }
                    )

                for tc in getattr(delta, "tool_calls", None) or []:
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
            input_usage = int(getattr(usage, "prompt_tokens", input_usage) or input_usage)
            output_usage = int(getattr(usage, "completion_tokens", output_usage) or output_usage)

    text = "".join(text_parts) or None
    tool_call_payload: Optional[dict] = None
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
        if tool_call_payload and re.fullmatch(
            r"\s*(```[\s\S]*```|<function_call>[\s\S]*</function_call>|<tool_call>[\s\S]*</tool_call>|<\w+>\s*\{[\s\S]*\}\s*</\w+>)\s*",
            text,
            flags=re.IGNORECASE,
        ):
            text = None

    if tool_call_payload and tool_call_payload.get("tool_call_name") == "message_to_user":
        parsed_args = tool_call_payload.get("tool_call_input") or {}
        text = parsed_args.get("message") or text
        tool_call_payload = None

    await stream_callback(
        {
            "message_uuid": stream_message_uuid,
            "role": "assistant",
            "content": text or "",
            "done": True,
            "usage": input_usage + output_usage,
        }
    )

    return LLMResponse(
        text=text,
        tool_call=tool_call_payload or {},
        stop_reason=finish_reason,
        input_usage=input_usage,
        output_usage=output_usage,
        model=model_name,
        message_uuid=stream_message_uuid,
    )


__all__ = ["consume_streaming_completion"]
