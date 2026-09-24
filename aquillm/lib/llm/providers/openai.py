"""OpenAI LLM interface."""

import asyncio as asyncio  # Preserve the provider's async dispatch patch seam.
import uuid
from os import getenv
from time import perf_counter
from typing import override

import structlog
from tiktoken import encoding_for_model

from lib.llm.optimizations.lm_lingua2_adapter import (
    maybe_compress_openai_style_messages,
)

from ..types.conversation import Conversation
from ..types.messages import AssistantMessage
from ..types.response import LLMResponse
from .base import LLMInterface
from .openai_context_policy import OpenAIContextPolicy
from .openai_request import prepare_request
from .openai_streaming import consume_streaming_completion
from .openai_tool_text import (
    decode_json_dict,
    extract_tool_call_from_text,
    is_textual_tool_call_only,
)
from .request_observability import (
    authorized_tool_name,
    current_correlation_id,
    current_stage,
    extract_usage,
    log_request_completed,
    new_correlation_id,
    safe_correlation_id,
    safe_stage,
)

try:
    from aquillm.settings import DEBUG
except ImportError:
    DEBUG = False

if DEBUG:
    from pprint import pp


gpt_enc = encoding_for_model("gpt-4o")
logger = structlog.stdlib.get_logger(__name__)


class OpenAIInterface(OpenAIContextPolicy, LLMInterface):
    """LLM interface for OpenAI models."""

    supports_request_observability = True

    @override
    def __init__(self, openai_client, model: str):
        self.client = openai_client
        self.base_args = {"model": model}

    @override
    async def get_message(self, *args, **kwargs) -> LLMResponse:
        synthesis_phase = kwargs.pop("_synthesis_phase", "initial")
        kwargs.pop("messages_pydantic", None)
        thinking_budget = kwargs.pop("thinking_budget", None)
        correlation_id = safe_correlation_id(
            kwargs.pop("_observability_correlation_id", None)
            or current_correlation_id()
            or new_correlation_id()
        )
        stage = safe_stage(
            kwargs.pop("_observability_stage", None)
            or current_stage()
            or "general_answer"
        )
        stream_callback = kwargs.pop("stream_callback", None)
        stream_message_uuid = str(
            kwargs.pop("stream_message_uuid", None) or uuid.uuid4()
        )
        system_text = kwargs.pop("system")
        message_list = kwargs.pop("messages")
        max_tokens = kwargs.pop("max_tokens")
        configured_max_tokens = max(0, int(max_tokens))
        tool_choice_raw = kwargs.pop("tool_choice", None)
        raw_tools = kwargs.get("tools")

        prepared = await prepare_request(
            self,
            system_text=system_text,
            message_list=message_list,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            tool_choice_raw=tool_choice_raw,
            kwargs=kwargs,
            compress_messages=maybe_compress_openai_style_messages,
        )
        arguments = prepared.arguments
        thinking_requested = prepared.thinking_requested

        request_timeout_s = float(getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "120"))
        try:
            max_request_timeout_s = float(
                getenv("OPENAI_REQUEST_TIMEOUT_MAX_SECONDS", "360")
            )
        except Exception:
            max_request_timeout_s = 360.0
        if max_request_timeout_s < request_timeout_s:
            max_request_timeout_s = request_timeout_s
        try:
            max_overflow_retries = int(getenv("OPENAI_CONTEXT_OVERFLOW_RETRIES", "3"))
        except Exception:
            max_overflow_retries = 3
        if max_overflow_retries < 3:
            max_overflow_retries = 3
        try:
            max_timeout_retries = int(getenv("OPENAI_TIMEOUT_RETRIES", "2"))
        except Exception:
            max_timeout_retries = 2
        if max_timeout_retries < 0:
            max_timeout_retries = 0

        stream_enabled = callable(stream_callback) and getenv(
            "OPENAI_STREAM_RESPONSES", "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        parsed_response: LLMResponse | None = None
        request_args = dict(arguments)
        if stream_enabled:
            request_args["stream"] = True
            request_args.setdefault("stream_options", {"include_usage": True})
        timeout_retries_used = 0
        max_total_retries = max_overflow_retries + max_timeout_retries
        for attempt in range(max_total_retries + 1):
            from lib.llm.evidence_guard import validate_request
            from lib.llm.synthesis_dispatch import dispatch

            validate_request(request_args, output_reserve=request_args["max_tokens"])
            request_started_at = perf_counter()
            try:
                if stream_enabled:
                    stream = await dispatch(
                        lambda: self.client.chat.completions.create(
                            timeout=request_timeout_s, **request_args
                        ),
                        output_reserve=request_args["max_tokens"],
                        payload=request_args,
                        provider="openai",
                        kind="transport_retry" if attempt else synthesis_phase,
                    )
                    parsed_response = await consume_streaming_completion(
                        stream=stream,
                        stream_callback=stream_callback,
                        stream_message_uuid=stream_message_uuid,
                        raw_tools=raw_tools,
                        model_name=self.base_args["model"],
                        correlation_id=correlation_id,
                        stage=stage,
                        configured_max_tokens=configured_max_tokens,
                        effective_max_tokens=max(
                            0,
                            int(request_args.get("max_tokens", 0) or 0),
                        ),
                        thinking_requested=thinking_requested,
                        request_started_at=request_started_at,
                    )
                else:
                    response = await dispatch(
                        lambda: self.client.chat.completions.create(
                            timeout=request_timeout_s, **request_args
                        ),
                        output_reserve=request_args["max_tokens"],
                        payload=request_args,
                        provider="openai",
                        kind="transport_retry" if attempt else synthesis_phase,
                    )
                    if DEBUG:
                        print("OpenAI SDK Response:")
                        pp(response)
                    text = response.choices[0].message.content
                    tool_call_payload: dict | None = None
                    raw_tool_call = (
                        response.choices[0].message.tool_calls[0]
                        if response.choices[0].message.tool_calls
                        else None
                    )
                    if raw_tool_call:
                        tool_call_payload = {
                            "tool_call_id": raw_tool_call.id or str(uuid.uuid4()),
                            "tool_call_name": raw_tool_call.function.name,
                            "tool_call_input": decode_json_dict(
                                raw_tool_call.function.arguments
                            ),
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

                    completed_at = perf_counter()
                    usage = extract_usage(getattr(response, "usage", None))

                    parsed_response = LLMResponse(
                        text=text,
                        tool_call=tool_call_payload or {},
                        stop_reason=response.choices[0].finish_reason,
                        input_usage=usage.prompt_tokens,
                        output_usage=usage.completion_tokens,
                        reasoning_usage=usage.reasoning_tokens,
                        model=self.base_args["model"],
                        message_uuid=stream_message_uuid,
                    )
                    log_request_completed(
                        logger=logger,
                        correlation_id=correlation_id,
                        stage=stage,
                        configured_max_tokens=configured_max_tokens,
                        effective_max_tokens=max(
                            0,
                            int(request_args.get("max_tokens", 0) or 0),
                        ),
                        thinking_requested=thinking_requested,
                        usage=usage,
                        finish_reason=response.choices[0].finish_reason,
                        request_started_at=request_started_at,
                        first_signal_at=None,
                        completed_at=completed_at,
                        tool_name=authorized_tool_name(tool_call_payload, raw_tools),
                    )
                break
            except Exception as exc:
                retry_args = self._retry_args_for_context_overflow(request_args, exc)
                if retry_args is not None and attempt < max_total_retries:
                    request_args = retry_args
                    continue

                if (
                    self._is_timeout_error(exc)
                    and timeout_retries_used < max_timeout_retries
                    and attempt < max_total_retries
                ):
                    timeout_retries_used += 1
                    timeout_retry_args = self._retry_args_for_timeout(
                        request_args,
                        timeout_retries_used,
                    )
                    if timeout_retry_args is not None:
                        request_args = timeout_retry_args
                    request_timeout_s = min(
                        max_request_timeout_s,
                        max(request_timeout_s, request_timeout_s * 1.5),
                    )
                    continue

                if attempt >= max_total_retries:
                    raise
                raise
        assert parsed_response is not None
        return parsed_response

    @override
    async def token_count(
        self, conversation: Conversation, new_message: str | None = None
    ) -> int:
        assistant_messages = [
            message for message in conversation if isinstance(message, AssistantMessage)
        ]
        if assistant_messages:
            return assistant_messages[-1].usage + (
                len(gpt_enc.encode(new_message)) if new_message else 0
            )
        return len(gpt_enc.encode(new_message)) if new_message else 0


__all__ = ["OpenAIInterface", "gpt_enc"]
