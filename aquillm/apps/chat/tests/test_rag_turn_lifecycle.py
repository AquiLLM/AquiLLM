"""Real tool executor and request-scope isolation."""

import asyncio
from threading import Event
from types import SimpleNamespace

import pytest

from lib.llm.turn_context import TurnContext, bind_turn, call_tool_async, current_turn
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.asyncio
async def test_both_tool_executor_hops_preserve_context():
    from aquillm.llm import llm_tool
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.types.messages import AssistantMessage

    @llm_tool(for_whom="assistant", required=[], param_descs={})
    def probe():
        """Inspect the bound request."""
        return {"result": str(id(current_turn().budget))}

    provider = OpenAIInterface(None, "test")
    message = AssistantMessage(
        content="",
        stop_reason="tool_use",
        tools=[probe],
        tool_call_id="one",
        tool_call_name="probe",
        tool_call_input={},
    )
    budget = TurnBudget(TurnLimits())
    with bind_turn(TurnContext(budget)):
        result = await call_tool_async(provider, message)
    assert result.result_dict["result"] == str(id(budget))
    assert current_turn() is None


@pytest.mark.asyncio
async def test_cancellation_during_sync_tool_wait_fences_late_work():
    from aquillm.llm import llm_tool
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.types.messages import AssistantMessage

    started, released = Event(), Event()
    writes = []
    budget = TurnBudget(TurnLimits())

    @llm_tool(for_whom="assistant", required=[], param_descs={})
    def blocked():
        """Wait for test release."""
        started.set()
        released.wait(3)
        budget.publish(lambda: writes.append("late"))
        return {"result": "late"}

    message = AssistantMessage(
        content="",
        stop_reason="tool_use",
        tools=[blocked],
        tool_call_id="one",
        tool_call_name="blocked",
        tool_call_input={},
    )
    with bind_turn(TurnContext(budget)):
        task = asyncio.create_task(
            call_tool_async(OpenAIInterface(None, "test"), message)
        )
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        released.set()
    assert not budget.can_publish()
    assert writes == []


@pytest.mark.asyncio
async def test_outer_scope_creates_distinct_turns_and_closes(monkeypatch):
    from apps.chat.services import rag_turn
    from apps.documents.services.source_loading import (
        SourceRuntime,
        current_source_runtime,
    )

    monkeypatch.setattr(
        rag_turn, "rag_preservation_config", lambda: SimpleNamespace(active=True)
    )
    monkeypatch.setattr(
        rag_turn, "create_runtime", lambda consumer, budget: SourceRuntime(budget, None)
    )
    consumer = SimpleNamespace()
    ids = []
    for _ in range(2):
        async with rag_turn.preservation_turn(consumer, max_func_calls=3) as runtime:
            ids.append(runtime.budget)
            assert current_turn().budget is current_source_runtime().budget
        assert not runtime.budget.can_publish()
        assert current_source_runtime() is current_turn() is None
    assert ids[0] is not ids[1]
