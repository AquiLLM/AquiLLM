"""Canned-response LLM provider for demos and offline UI walkthroughs.

Initialised with a list of pre-scripted responses (strings or LLMResponse
objects). Each call to get_message consumes the next entry. Plain strings
become end_turn text responses; full LLMResponse objects pass through, which
is how scripted tool calls are emitted.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Optional, Union, override

from ..types.conversation import Conversation
from ..types.response import LLMResponse
from .base import LLMInterface


ScriptStep = Union[str, LLMResponse]


def tool_call_response(
    name: str,
    arguments: dict,
    text: Optional[str] = None,
    call_id: Optional[str] = None,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_call={
            "tool_call_id": call_id or uuid.uuid4().hex,
            "tool_call_name": name,
            "tool_call_input": arguments,
        },
        stop_reason="tool_use",
        input_usage=0,
        output_usage=0,
        model="fake",
    )


class FakeInterface(LLMInterface):
    base_args: dict = {"model": "fake"}

    @override
    def __init__(
        self,
        responses: list[ScriptStep],
        client=None,
        simulate_streaming: bool = True,
        chunk_size: int = 20,
        chunk_delay_s: float = 0.04,
    ):
        if not responses:
            raise ValueError("FakeInterface requires at least one canned response")
        self.responses = responses
        self.client = client
        self.simulate_streaming = simulate_streaming
        self.chunk_size = max(1, chunk_size)
        self.chunk_delay_s = max(0.0, chunk_delay_s)
        self._index = 0

    def _coerce(self, step: ScriptStep, message_uuid: str) -> LLMResponse:
        if isinstance(step, LLMResponse):
            response = step.model_copy()
            if not response.message_uuid:
                response.message_uuid = message_uuid
            if not response.model:
                response.model = "fake"
            return response
        return LLMResponse(
            text=step,
            tool_call={},
            stop_reason="end_turn",
            input_usage=0,
            output_usage=0,
            model="fake",
            message_uuid=message_uuid,
        )

    @override
    async def get_message(self, *args, **kwargs) -> LLMResponse:
        # Out-of-band internal calls (e.g. WSConversation.set_name auto-titling) hit
        # this same interface. They are not part of the scripted chat flow, so do
        # not consume an index — serve a canned title-style response and return.
        if self._is_internal_title_call(kwargs):
            return LLMResponse(
                text="Demo Conversation",
                tool_call={},
                stop_reason="end_turn",
                input_usage=0,
                output_usage=0,
                model="fake",
            )

        index = min(self._index, len(self.responses) - 1)
        self._index += 1

        stream_message_uuid = kwargs.get("stream_message_uuid") or uuid.uuid4().hex
        response = self._coerce(self.responses[index], stream_message_uuid)

        stream_callback = kwargs.get("stream_callback")
        text = response.text or ""
        has_tool_call = bool(response.tool_call)

        if (
            self.simulate_streaming
            and callable(stream_callback)
            and text
            and not has_tool_call
        ):
            accumulated = ""
            for start in range(0, len(text), self.chunk_size):
                accumulated = text[: start + self.chunk_size]
                await stream_callback(
                    {
                        "message_uuid": response.message_uuid,
                        "role": "assistant",
                        "content": accumulated,
                        "done": False,
                    }
                )
                if self.chunk_delay_s:
                    await asyncio.sleep(self.chunk_delay_s)
            await stream_callback(
                {
                    "message_uuid": response.message_uuid,
                    "role": "assistant",
                    "content": text,
                    "done": True,
                    "stop_reason": response.stop_reason,
                    "usage": response.input_usage + response.output_usage,
                }
            )

        return response

    @staticmethod
    def _is_internal_title_call(kwargs: dict) -> bool:
        system = str(kwargs.get("system") or "")
        return "title" in system.lower() and "conversation" in system.lower()

    @override
    async def token_count(
        self, conversation: Conversation, new_message: Optional[str] = None
    ) -> int:
        return 0


__all__ = ["FakeInterface", "tool_call_response"]
