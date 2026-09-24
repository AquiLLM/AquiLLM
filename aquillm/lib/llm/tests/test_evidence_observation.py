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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_event", ["sdk_start", "sdk_end", "ledger"])
async def test_failed_observer_preserves_committed_returns_and_exceptions(
    failure_event,
):
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    def sink(event, data):
        if event == failure_event:
            raise RuntimeError("observer failed")

    async def success():
        return "Yes"

    error = ValueError("original SDK failure")

    async def failure():
        raise error

    with observe(sink) as state:
        budget = TurnBudget(TurnLimits())
        assert budget.reserve_action("query")
        assert budget.actions_used == 1
        assert await dispatch(success, output_reserve=1) == "Yes"
        with pytest.raises(ValueError) as raised:
            await dispatch(failure, output_reserve=1)
        assert raised.value is error
        assert state.failed.is_set()


@pytest.mark.asyncio
async def test_observer_copy_failure_and_concurrent_turns_are_isolated():
    import asyncio

    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("copy failed")

    async def turn(broken):
        with observe(lambda *args: None) as state:
            await asyncio.sleep(0)
            publish("sdk_start", {"value": Uncopyable()} if broken else {})
            publish("turn_complete", {})
            assert state.completed.is_set() and state.sdk_started.is_set()
            return state.failed.is_set()

    assert await asyncio.gather(turn(True), turn(False)) == [True, False]
