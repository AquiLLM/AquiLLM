"""Retrieval-specific deadlines preserve stricter policy and unrelated tools."""

import asyncio
from threading import Event

import pytest
from channels.db import database_sync_to_async

from lib.llm.turn_context import TurnContext, bind_turn, call_tool_async
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.asyncio
async def test_targeted_action_keeps_stricter_tool_timeout_and_fences_late_result(
    monkeypatch,
):
    from apps.chat.services.rag_acquisition import acquire_evidence
    from apps.chat.services.rag_coverage import AcquisitionAction, CoverageAssessment
    from lib.retrieval.evidence import SourceEvidence

    monkeypatch.setenv("TOOL_CALL_TIMEOUT_SECONDS", "0.05")
    budget = TurnBudget(TurnLimits())
    budget.reserve_action("initial model tool")
    released, finished = Event(), Event()
    late = []

    def worker():
        released.wait(2)
        budget.publish(lambda: late.append(True))
        finished.set()
        return {"result": []}

    async def execute(action):
        budget.reserve_action(action.signature)
        return await database_sync_to_async(worker, thread_sensitive=False)()

    async def assess(*args, **kwargs):
        return CoverageAssessment(
            ("yield",),
            (),
            ("yield",),
            AcquisitionAction("vector", "yield", aspect="yield"),
        )

    with bind_turn(TurnContext(budget)):
        task = asyncio.create_task(
            acquire_evidence(
                "Explain yield",
                budget=budget,
                llm=None,
                execute=execute,
                initial_results=({"result": []},),
                initial_acquired=True,
                iterative=True,
                evidence_views=lambda _: (SourceEvidence(1, "doc", 0, "r", "method"),),
                assess=assess,
            )
        )
        try:
            done, _ = await asyncio.wait({task}, timeout=0.3)
            assert task in done, "refinement exceeded the stricter tool timeout"
            assert task.result().stop_reason == "partial_unknown"
        finally:
            released.set()
            await task
        assert await asyncio.to_thread(finished.wait, 1)
        assert late == []
        assert budget.actions_used == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_non_source_tool_after_retrieval_expiry_keeps_original_policy(cancel):
    from aquillm.llm import llm_tool
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.types.messages import AssistantMessage

    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    now[0] = 16.0
    called = []

    @llm_tool(for_whom="assistant", required=[], param_descs={})
    def calculator():
        """Run an unrelated local computation."""
        called.append(True)
        return {"result": 4}

    message = AssistantMessage(
        content="",
        stop_reason="tool_use",
        tools=[calculator],
        tool_call_id="local",
        tool_call_name="calculator",
        tool_call_input={},
    )
    with bind_turn(TurnContext(budget)):
        if cancel:
            budget.close("cancelled")
            with pytest.raises(asyncio.CancelledError):
                await call_tool_async(OpenAIInterface(None, "test"), message)
            assert not called
        else:
            result = await call_tool_async(OpenAIInterface(None, "test"), message)
            assert result.result_dict == {"result": 4}
            assert called == [True]
        assert not budget.reserve_action("source retrieval")
