"""Integration tests for adaptive tool budget behavior in LLMInterface.spin()."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lib.llm.providers.base import LLMInterface
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage


class _DummyLLMInterface(LLMInterface):
    def __init__(self, client=None):
        self.client = client
        self.base_args = {}

    async def get_message(self, *args, **kwargs):
        raise NotImplementedError

    async def token_count(self, conversation: Conversation, new_message: str | None = None) -> int:
        return 0


def _assistant_tool_call(call_number: int, query: str = "alpha") -> AssistantMessage:
    return AssistantMessage(
        content=f"tool call {call_number}",
        stop_reason="tool_use",
        tool_call_id=f"tc-{call_number}",
        tool_call_name="vector_search",
        tool_call_input={"q": query},
    )


def _assistant_final_answer(text: str = "final synthesis answer") -> AssistantMessage:
    return AssistantMessage(content=text, stop_reason="stop")


def _tool_result(call_number: int, *, exception: bool = False) -> ToolMessage:
    result_dict = {"exception": "Tool call timed out"} if exception else {"result": [{"id": call_number}]}
    return ToolMessage(
        content=str(result_dict),
        tool_name="vector_search",
        for_whom="assistant",
        result_dict=result_dict,
    )


def _convo_with_n_calls(
    n_calls: int,
    *,
    same_query: bool = True,
    exception_results: bool = False,
) -> Conversation:
    messages: list = [UserMessage(content="Find relevant sources.")]
    for i in range(1, n_calls + 1):
        if i > 1:
            messages.append(_tool_result(i - 1, exception=exception_results))
        query = "same-query" if same_query else f"query-{i}"
        messages.append(_assistant_tool_call(i, query=query))
    return Conversation(system="sys", messages=messages)


@pytest.mark.asyncio
async def test_spin_allows_more_calls_with_per_tool_override(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CALLS_PER_TOOL_NAME", "2")
    monkeypatch.setenv("LLM_TOOL_CALL_LIMITS", "vector_search:4")
    monkeypatch.setenv("LLM_REPEAT_TOOL_BREAK_THRESHOLD", "99")
    monkeypatch.setenv("LLM_TOOL_NO_PROGRESS_BREAK_THRESHOLD", "99")
    monkeypatch.delenv("LLM_TOOL_BUDGET_UNITS_PER_TURN", raising=False)
    monkeypatch.delenv("LLM_TOOL_COST_WEIGHTS", raising=False)

    llm = _DummyLLMInterface()
    llm.complete = AsyncMock(
        side_effect=[
            (_convo_with_n_calls(1), "changed"),
            (_convo_with_n_calls(2), "changed"),
            (_convo_with_n_calls(3), "changed"),
            (_convo_with_n_calls(4), "changed"),
            (Conversation(system="sys", messages=[UserMessage(content="q"), _assistant_final_answer()]), "changed"),
            (Conversation(system="sys", messages=[UserMessage(content="q"), _assistant_final_answer()]), "unchanged"),
        ]
    )
    send_func = AsyncMock()

    await llm.spin(
        convo=Conversation(system="sys", messages=[UserMessage(content="start")]),
        max_func_calls=10,
        send_func=send_func,
        max_tokens=256,
    )

    assert llm.complete.await_count == 6


@pytest.mark.asyncio
async def test_spin_breaks_when_configured_per_tool_limit_is_reached(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CALLS_PER_TOOL_NAME", "2")
    monkeypatch.setenv("LLM_TOOL_CALL_LIMITS", "vector_search:3")
    monkeypatch.setenv("LLM_REPEAT_TOOL_BREAK_THRESHOLD", "99")
    monkeypatch.setenv("LLM_TOOL_NO_PROGRESS_BREAK_THRESHOLD", "99")
    monkeypatch.delenv("LLM_TOOL_BUDGET_UNITS_PER_TURN", raising=False)
    monkeypatch.delenv("LLM_TOOL_COST_WEIGHTS", raising=False)

    llm = _DummyLLMInterface()
    llm.complete = AsyncMock(
        side_effect=[
            (_convo_with_n_calls(1), "changed"),
            (_convo_with_n_calls(2), "changed"),
            (_convo_with_n_calls(3), "changed"),
            (Conversation(system="sys", messages=[UserMessage(content="q"), _assistant_final_answer()]), "changed"),
            (Conversation(system="sys", messages=[UserMessage(content="q"), _assistant_final_answer()]), "unchanged"),
        ]
    )
    send_func = AsyncMock()

    await llm.spin(
        convo=Conversation(system="sys", messages=[UserMessage(content="start")]),
        max_func_calls=10,
        send_func=send_func,
        max_tokens=256,
    )

    assert llm.complete.await_count == 5


@pytest.mark.asyncio
async def test_spin_breaks_on_repeated_no_progress_below_count_ceiling(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CALLS_PER_TOOL_NAME", "10")
    monkeypatch.setenv("LLM_REPEAT_TOOL_BREAK_THRESHOLD", "99")
    monkeypatch.setenv("LLM_TOOL_NO_PROGRESS_BREAK_THRESHOLD", "2")
    monkeypatch.delenv("LLM_TOOL_CALL_LIMITS", raising=False)
    monkeypatch.delenv("LLM_TOOL_BUDGET_UNITS_PER_TURN", raising=False)
    monkeypatch.delenv("LLM_TOOL_COST_WEIGHTS", raising=False)

    llm = _DummyLLMInterface()
    llm.complete = AsyncMock(
        side_effect=[
            (_convo_with_n_calls(1, same_query=False, exception_results=False), "changed"),
            (_convo_with_n_calls(2, same_query=False, exception_results=True), "changed"),
            (_convo_with_n_calls(3, same_query=False, exception_results=True), "changed"),
            (Conversation(system="sys", messages=[UserMessage(content="q"), _assistant_final_answer()]), "changed"),
            (Conversation(system="sys", messages=[UserMessage(content="q"), _assistant_final_answer()]), "unchanged"),
        ]
    )
    send_func = AsyncMock()

    await llm.spin(
        convo=Conversation(system="sys", messages=[UserMessage(content="start")]),
        max_func_calls=10,
        send_func=send_func,
        max_tokens=256,
    )

    assert llm.complete.await_count == 5


@pytest.mark.asyncio
async def test_spin_breaks_when_weighted_budget_units_are_exhausted(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CALLS_PER_TOOL_NAME", "10")
    monkeypatch.setenv("LLM_REPEAT_TOOL_BREAK_THRESHOLD", "99")
    monkeypatch.setenv("LLM_TOOL_NO_PROGRESS_BREAK_THRESHOLD", "99")
    monkeypatch.setenv("LLM_TOOL_BUDGET_UNITS_PER_TURN", "3")
    monkeypatch.setenv("LLM_TOOL_COST_WEIGHTS", "vector_search:2")
    monkeypatch.delenv("LLM_TOOL_CALL_LIMITS", raising=False)

    llm = _DummyLLMInterface()
    llm.complete = AsyncMock(
        side_effect=[
            (_convo_with_n_calls(1, same_query=False), "changed"),
            (_convo_with_n_calls(2, same_query=False), "changed"),
            (Conversation(system="sys", messages=[UserMessage(content="q"), _assistant_final_answer()]), "changed"),
            (Conversation(system="sys", messages=[UserMessage(content="q"), _assistant_final_answer()]), "unchanged"),
        ]
    )
    send_func = AsyncMock()

    await llm.spin(
        convo=Conversation(system="sys", messages=[UserMessage(content="start")]),
        max_func_calls=10,
        send_func=send_func,
        max_tokens=256,
    )

    assert llm.complete.await_count == 4


@pytest.mark.asyncio
async def test_spin_preserves_final_synthesis_step_after_break(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CALLS_PER_TOOL_NAME", "2")
    monkeypatch.setenv("LLM_REPEAT_TOOL_BREAK_THRESHOLD", "99")
    monkeypatch.setenv("LLM_TOOL_NO_PROGRESS_BREAK_THRESHOLD", "99")
    monkeypatch.delenv("LLM_TOOL_CALL_LIMITS", raising=False)
    monkeypatch.delenv("LLM_TOOL_BUDGET_UNITS_PER_TURN", raising=False)
    monkeypatch.delenv("LLM_TOOL_COST_WEIGHTS", raising=False)

    llm = _DummyLLMInterface()
    final_convo = Conversation(
        system="sys",
        messages=[UserMessage(content="q"), _assistant_final_answer("synthesized response")],
    )
    llm.complete = AsyncMock(
        side_effect=[
            (_convo_with_n_calls(1), "changed"),
            (_convo_with_n_calls(2), "changed"),
            (final_convo, "changed"),
            (final_convo, "unchanged"),
        ]
    )
    send_func = AsyncMock()

    await llm.spin(
        convo=Conversation(system="sys", messages=[UserMessage(content="start")]),
        max_func_calls=10,
        send_func=send_func,
        max_tokens=256,
    )

    final_sent_convo = send_func.await_args_list[-1].args[0]
    assert isinstance(final_sent_convo[-1], AssistantMessage)
    assert final_sent_convo[-1].content == "synthesized response"

