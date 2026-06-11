"""Tests for complete_turn deterministic fallback extended to tool_choice=auto (Task 6)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse
from lib.llm.types.tools import ToolChoice
from aquillm.llm import llm_tool


@llm_tool(
    for_whom="assistant",
    required=["search_string"],
    param_descs={"search_string": "query text", "top_k": "number of results"},
)
def vector_search(search_string: str, top_k: int = 10) -> dict:
    """Search documents by semantic similarity."""
    return {"result": []}


@llm_tool(for_whom="assistant", required=[], param_descs={})
def _fake_other_tool() -> dict:
    """Some other non-search tool."""
    return {}


def _user_msg_with_tool(content: str, tool: object, choice_type: str) -> UserMessage:
    return UserMessage(
        content=content,
        tools=[tool],
        tool_choice=ToolChoice(type=choice_type),
    )


_vector_search_tool = vector_search
_other_tool = _fake_other_tool


class _FakeLLM:
    def __init__(self, response: LLMResponse):
        self._response = response
        self.base_args = {}
        self.calls = 0

    async def get_message(self, *args, **kwargs):
        self.calls += 1
        return self._response

    def call_tool(self, msg):
        raise AssertionError("call_tool should not run in these tests")


@pytest.mark.asyncio
async def test_deterministic_fallback_fires_for_auto_with_vector_search():
    """tool_choice=auto + vector_search + empty LLM text → deterministic tool call injected."""
    empty_response = LLMResponse(
        text="",
        tool_call=None,
        stop_reason="end_turn",
        input_usage=1,
        output_usage=1,
    )
    llm = _FakeLLM(empty_response)
    user_msg = _user_msg_with_tool(
        "What does this paper say about spectral analysis?",
        _vector_search_tool,
        "auto",
    )
    convo = Conversation(system="sys", messages=[user_msg])

    result_convo, status = await complete_conversation_turn(llm, convo, max_tokens=512)

    assert status == "changed"
    last = result_convo[-1]
    assert isinstance(last, AssistantMessage)
    assert last.tool_call_name == "vector_search"
    assert last.tool_call_id is not None


@pytest.mark.asyncio
async def test_deterministic_fallback_for_auto_uses_user_message_content_as_query():
    """The injected vector_search call uses the user message as search_string."""
    empty_response = LLMResponse(
        text="",
        tool_call=None,
        stop_reason="end_turn",
        input_usage=1,
        output_usage=1,
    )
    llm = _FakeLLM(empty_response)
    user_msg = _user_msg_with_tool(
        "Explain the Lyman-alpha forest",
        _vector_search_tool,
        "auto",
    )
    convo = Conversation(system="sys", messages=[user_msg])

    result_convo, _ = await complete_conversation_turn(llm, convo, max_tokens=512)

    last = result_convo[-1]
    assert isinstance(last, AssistantMessage)
    assert last.tool_call_input is not None
    assert "Lyman-alpha forest" in last.tool_call_input.get("search_string", "")


@pytest.mark.asyncio
async def test_deterministic_fallback_not_injected_when_auto_has_non_search_tool_only():
    """tool_choice=auto with a non-vector_search tool does NOT inject a deterministic call."""
    empty_response = LLMResponse(
        text="",
        tool_call=None,
        stop_reason="end_turn",
        input_usage=1,
        output_usage=1,
    )
    llm = _FakeLLM(empty_response)
    user_msg = _user_msg_with_tool(
        "Run the processing pipeline.",
        _other_tool,
        "auto",
    )
    convo = Conversation(system="sys", messages=[user_msg])

    result_convo, status = await complete_conversation_turn(llm, convo, max_tokens=512)

    last = result_convo[-1]
    assert isinstance(last, AssistantMessage)
    assert last.tool_call_name != "vector_search"


@pytest.mark.asyncio
async def test_deterministic_fallback_not_injected_when_llm_produces_text():
    """When the LLM provides usable text, no deterministic injection happens."""
    text_response = LLMResponse(
        text="Here is my answer about spectral analysis.",
        tool_call=None,
        stop_reason="end_turn",
        input_usage=1,
        output_usage=1,
    )
    llm = _FakeLLM(text_response)
    user_msg = _user_msg_with_tool(
        "What does this paper say about spectral analysis?",
        _vector_search_tool,
        "auto",
    )
    convo = Conversation(system="sys", messages=[user_msg])

    result_convo, _ = await complete_conversation_turn(llm, convo, max_tokens=512)

    last = result_convo[-1]
    assert isinstance(last, AssistantMessage)
    assert last.tool_call_name is None
    assert "spectral analysis" in (last.content or "")
