"""Mandatory acquisition shares the original deadline and preserves reserves."""

import asyncio
from contextvars import copy_context
from types import SimpleNamespace

import pytest
import pytest_asyncio

from apps.chat.services.rag_acquisition import acquire_evidence
from apps.chat.services.retrieval_query_proposals import build_query_proposals
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.llm.turn_context import TurnContext, bind_turn
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest_asyncio.fixture
async def clock(monkeypatch):
    # Advance the actual loop timer and ledger together, so asyncio.wait_for
    # really cancels overdue work without sleeping for seconds in each case.
    now = [0.0]
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "time", lambda: now[0])

    async def elapse(seconds):
        for _ in range(int(seconds * 4)):
            now[0] += 0.25
            for _ in range(3):
                await asyncio.sleep(0)

    return now, elapse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spent, duration, tool_seconds, succeeds, remaining, dispatched",
    [
        (2, 4, "10", True, 9000, True),
        (0, 4, "2", False, 13000, True),
        (0, 12, "10", False, 5000, True),
        (6, 8, "10", False, 4000, True),
        (11, 1, "10", False, 4000, False),
    ],
)
async def test_initial_search_uses_remaining_allowance_without_spending_reserves(
    monkeypatch, clock, spent, duration, tool_seconds, succeeds, remaining, dispatched,
):
    now, elapse = clock
    monkeypatch.setenv("TOOL_CALL_TIMEOUT_SECONDS", tool_seconds)
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    now[0] = spent
    contexts = []

    async def execute(action):
        contexts.append(copy_context())
        assert budget.reserve_action(action.signature)
        await elapse(duration)
        return {"result": ["available evidence"]}

    with bind_turn(TurnContext(budget)):
        result = await acquire_evidence(
            "Find the calibration", budget=budget, llm=None,
            execute=execute, evidence_views=lambda _: (),
        )
    assert bool(result.results) is succeeds
    assert bool(contexts) is dispatched
    assert budget.actions_used == int(dispatched)
    assert budget.remaining_ms() == remaining
    assert budget.limits.retrieval_ms == 15000
    if not succeeds:
        assert result.stop_reason == "partial_unknown"
    # An ended operation must never publish a late worker result, even though
    # the original turn remains open for scoring and completion.
    for context in contexts:
        assert context.run(budget.publish, lambda: None) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_duration, actions", [(4, 2), (7, 1)])
async def test_prepared_original_gets_mandatory_time_but_added_query_keeps_cap(
    monkeypatch, clock, initial_duration, actions,
):
    from apps.chat.services import retrieval_query_runtime as query_runtime

    now, elapse = clock
    monkeypatch.setenv("TOOL_CALL_TIMEOUT_SECONDS", "10")
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    now[0] = 2
    runtime = SourceRuntime(budget, SimpleNamespace())
    rows = build_query_proposals("Find calibration", rewrite="Calibration details")
    runtime._query_proposals = rows

    async def database_validation(function, bound):
        # Authorization DB work is outside this pure timing regression. Keep
        # the real prepared-query membership and proposal-scope adapter.
        assert function is query_runtime.validate_bound
        assert bound.runtime is runtime

    monkeypatch.setattr(query_runtime, "_database_call", database_validation)
    seen = []

    async def execute(action):
        seen.append(action.query)
        assert budget.reserve_action(action.signature)
        await elapse(initial_duration if len(seen) == 1 else 4)
        return {"result": [action.query]}

    with source_runtime_scope(runtime), bind_turn(TurnContext(budget)):
        result = await acquire_evidence(
            rows[0].text, budget=budget, llm=None, execute=execute,
            evidence_views=lambda _: (), query_proposals=rows,
        )
    assert seen == [row.text for row in rows[:actions]]
    assert result.results == ({"result": [rows[0].text]},)
    assert result.stop_reason == ("partial_unknown" if actions == 2 else "initial")
    assert budget.actions_used == actions
    assert budget.remaining_ms() == 6000
