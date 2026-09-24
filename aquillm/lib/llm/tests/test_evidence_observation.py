"""Observation is opt-in, scoped and sees actual shaped SDK dispatches."""

import pytest

from lib.evidence_observation import observe, publish
from lib.llm.synthesis_dispatch import dispatch


@pytest.mark.asyncio
async def test_sdk_observer_is_scoped_and_does_not_change_request():
    events = []
    request = {"messages": [{"content": "source μ tail"}]}

    async def sdk():
        return "answer"

    with observe(lambda name, value: events.append((name, value))):
        assert (
            await dispatch(sdk, output_reserve=30, payload=request, provider="openai")
            == "answer"
        )
    publish("outside", {})
    assert [x[0] for x in events] == ["sdk_start", "sdk_end"]
    assert events[0][1]["payload"] == request
    request["messages"].clear()
    assert events[0][1]["payload"]["messages"]


@pytest.mark.asyncio
async def test_denied_dispatch_is_not_counted_as_actual_sdk_work():
    import asyncio

    from lib.llm.turn_context import TurnContext, bind_turn
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    budget = TurnBudget(TurnLimits())
    budget.close("cancelled")
    events = []
    with observe(lambda *event: events.append(event)), bind_turn(TurnContext(budget)):
        with pytest.raises(asyncio.CancelledError):
            await dispatch(lambda: None, output_reserve=1)
    assert not events
