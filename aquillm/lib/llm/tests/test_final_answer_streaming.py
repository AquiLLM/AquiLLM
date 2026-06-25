"""Regression tests for final-answer-only streaming."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse


@pytest.mark.asyncio
async def test_post_tool_stream_replays_only_final_retry_answer():
    stream_payloads: list[dict] = []

    async def _capture_stream(payload: dict) -> None:
        stream_payloads.append(payload)

    async def _fake_get_message(**kwargs):
        callback = kwargs.get("stream_callback")
        if callable(callback):
            await callback(
                {
                    "message_uuid": kwargs.get("stream_message_uuid"),
                    "role": "assistant",
                    "content": "I'll help explain this after retrieval.",
                    "done": True,
                }
            )
        if len(llm.get_message.await_args_list) == 1:
            return LLMResponse(
                text="I'll help explain this after retrieval.",
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
                model="fake",
            )
        return LLMResponse(
            text="Final synthesized answer with supporting detail.",
            tool_call={},
            stop_reason="stop",
            input_usage=1,
            output_usage=1,
            model="fake",
        )

    llm = SimpleNamespace(base_args={}, get_message=AsyncMock(side_effect=_fake_get_message))
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Explain this paper."),
            ToolMessage(
                content="Retrieved document text.",
                tool_name="whole_document",
                for_whom="assistant",
                result_dict={"result": "Retrieved document text."},
            ),
        ],
    )

    updated, changed = await complete_conversation_turn(
        llm,
        convo,
        max_tokens=1024,
        stream_func=_capture_stream,
    )

    assert changed == "changed"
    assert updated[-1].content == "Final synthesized answer with supporting detail."
    assert [payload["content"] for payload in stream_payloads] == [
        "Final synthesized answer with supporting detail."
    ]
    assert stream_payloads[0]["done"] is True


@pytest.mark.asyncio
async def test_final_only_stream_appends_sources_after_validation():
    stream_payloads: list[dict] = []

    async def _capture_stream(payload: dict) -> None:
        stream_payloads.append(payload)

    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            return_value=LLMResponse(
                text="Final cited answer [doc:doc-a chunk:7].",
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
                model="fake",
            )
        ),
    )
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Summarize this source."),
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={"result": [{"chunk_id": 7, "doc_id": "doc-a", "text": "alpha"}]},
            ),
        ],
    )

    updated, changed = await complete_conversation_turn(
        llm,
        convo,
        max_tokens=1024,
        stream_func=_capture_stream,
    )

    assert changed == "changed"
    assert "Sources:\n- [doc:doc-a chunk:7]" in updated[-1].content
    assert stream_payloads == [
        {
            "message_uuid": str(updated[-1].message_uuid),
            "role": "assistant",
            "content": updated[-1].content,
            "done": True,
            "stop_reason": "stop",
            "usage": 2,
        }
    ]
