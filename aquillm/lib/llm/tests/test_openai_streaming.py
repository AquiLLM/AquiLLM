"""Tests for OpenAI-compatible streaming visibility rules."""
from __future__ import annotations

from types import SimpleNamespace

from lib.llm.providers.openai_streaming import consume_streaming_completion


class _FakeStream:
    def __init__(self, chunks: list[SimpleNamespace]):
        self._chunks = chunks

    def __aiter__(self):
        self._iterator = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration


def _chunk(content: str | None, finish_reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=[]),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


async def test_streaming_textual_tool_call_is_not_sent_to_visible_stream():
    payloads: list[dict] = []

    async def _capture(payload: dict) -> None:
        payloads.append(payload)

    tool_text = (
        '<tool_call>{"name":"vector_search",'
        '"arguments":{"search_string":"memory"}}</tool_call>'
    )

    response = await consume_streaming_completion(
        stream=_FakeStream([_chunk(tool_text[:38]), _chunk(tool_text[38:], "stop")]),
        stream_callback=_capture,
        stream_message_uuid="msg-1",
        raw_tools=[{"name": "vector_search"}],
        model_name="test-model",
    )

    assert response.text is None
    assert response.tool_call["tool_call_name"] == "vector_search"
    assert all("<tool_call>" not in payload.get("content", "") for payload in payloads)
    assert all("vector_search" not in payload.get("content", "") for payload in payloads)


async def test_streaming_raw_tool_transcript_is_suppressed_when_tools_are_available():
    payloads: list[dict] = []

    async def _capture(payload: dict) -> None:
        payloads.append(payload)

    raw_transcript = (
        "I'll help you understand the mathematical content. "
        "Let me first retrieve the main document.\n\n"
        "Tool:retrieve\n\n"
        '{"document_ids":["doc-1"],"query":"mathematical formulas"}\n\n'
        "Sources:\n- [doc:doc-1]"
    )

    response = await consume_streaming_completion(
        stream=_FakeStream([_chunk(raw_transcript[:70]), _chunk(raw_transcript[70:], "stop")]),
        stream_callback=_capture,
        stream_message_uuid="msg-2",
        raw_tools=[{"name": "whole_document"}],
        model_name="test-model",
    )

    assert response.text == raw_transcript
    assert payloads == []
    assert all("Tool:retrieve" not in payload.get("content", "") for payload in payloads)
    assert all("I'll help" not in payload.get("content", "") for payload in payloads)


async def test_streaming_think_blocks_are_removed_from_visible_stream():
    payloads: list[dict] = []

    async def _capture(payload: dict) -> None:
        payloads.append(payload)

    response = await consume_streaming_completion(
        stream=_FakeStream([
            _chunk("<think>I should inspect this first.</think>"),
            _chunk("Final answer.", "stop"),
        ]),
        stream_callback=_capture,
        stream_message_uuid="msg-3",
        raw_tools=None,
        model_name="test-model",
    )

    assert response.text == "Final answer."
    assert payloads[-1]["content"] == "Final answer."
    assert all("<think>" not in payload.get("content", "") for payload in payloads)
    assert all("inspect this first" not in payload.get("content", "") for payload in payloads)


async def test_streaming_tool_code_fragment_is_suppressed():
    payloads: list[dict] = []

    async def _capture(payload: dict) -> None:
        payloads.append(payload)

    await consume_streaming_completion(
        stream=_FakeStream([_chunk("<tool_code> Tool"), _chunk("Real answer paragraph.", "stop")]),
        stream_callback=_capture,
        stream_message_uuid="msg-tool-code",
        raw_tools=[{"name": "whole_document"}],
        model_name="test-model",
    )

    assert all("<tool_code>" not in payload.get("content", "") for payload in payloads)
    assert all(payload.get("content", "") != " Tool" for payload in payloads)


async def test_streaming_deferred_retrieval_phrase_is_suppressed_without_tools():
    payloads: list[dict] = []

    async def _capture(payload: dict) -> None:
        payloads.append(payload)

    response = await consume_streaming_completion(
        stream=_FakeStream([_chunk("I'll retrieve the passage now.", "stop")]),
        stream_callback=_capture,
        stream_message_uuid="msg-4",
        raw_tools=None,
        model_name="test-model",
    )

    assert response.text == "I'll retrieve the passage now."
    assert payloads == []
