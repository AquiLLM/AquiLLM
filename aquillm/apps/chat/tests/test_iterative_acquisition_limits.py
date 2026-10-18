"""Regressions for real follow-up capacity on one cumulative retrieval ledger."""

import asyncio
from types import SimpleNamespace

import pytest

from apps.chat.services.rag_acquisition import acquire_evidence
from apps.chat.services.rag_coverage import AcquisitionAction, CoverageAssessment
from lib.retrieval.evidence import SourceEvidence, fingerprint_source
from lib.retrieval.operation import _OPERATION, operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def source(number):
    text = f"Unresolved evidence {number}."
    return SourceEvidence(number, "doc", number, fingerprint_source(text), text)


def views(results):
    return tuple(s for result in results for s in result["sources"])


async def planner(*args, **kwargs):
    return CoverageAssessment(
        ("missing method",), (), ("missing method",),
        AcquisitionAction("vector", "targeted method", aspect="missing method"),
        "assessed",
    )


def test_source_ceiling_reserves_capacity_without_resetting_shared_ledger():
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    with operation_scope(budget, source_limit=30):
        assert all(budget.admit_source(source(i).identity) for i in range(30))
        assert not budget.admit_source(source(30).identity)
        assert budget.admit_source(source(0).identity)
        with operation_scope(budget, source_limit=45):
            assert not budget.admit_source(source(30).identity)
    with operation_scope(budget, source_limit=45):
        assert all(budget.admit_source(source(i).identity) for i in range(30, 45))
        assert not budget.admit_source(source(45).identity)
    assert budget.sources_used == 45


def test_captured_source_ceiling_cannot_be_mutated():
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    with operation_scope(budget, source_limit=30) as operation:
        with pytest.raises(AttributeError):
            operation.source_limit = 45
        assert budget.remaining_source_capacity() == 30


def test_nested_source_only_scope_cannot_replace_original_budget():
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    with operation_scope(budget, source_limit=30):
        with operation_scope(budget):
            with pytest.raises(ValueError, match="budget identity"):
                with operation_scope(TurnBudget(TurnLimits(), clock=lambda: 0.0)):
                    pytest.fail("nested caller escaped original source ceiling")


@pytest.mark.asyncio
async def test_model_tool_first_search_reserves_capacity_before_handoff():
    from lib.llm.turn_context import TurnContext, bind_turn, call_tool_async

    budget = TurnBudget(TurnLimits())
    state = TurnContext(budget, initial_source_limit=30)
    message = SimpleNamespace(tool_call_name="vector_search", tool_call_input={})

    def call_tool(message):
        number = budget.actions_used
        assert budget.reserve_action(str(number))
        return [
            source(i) for i in range(number * 45, (number + 1) * 45)
            if budget.admit_source(source(i).identity)
        ]

    with bind_turn(state):
        first = await call_tool_async(SimpleNamespace(call_tool=call_tool), message)
        second = await call_tool_async(SimpleNamespace(call_tool=call_tool), message)
    assert len(first) == 30
    assert len(second) == 15
    assert budget.sources_used == 45


@pytest.mark.asyncio
async def test_copied_worker_context_retains_ceiling_and_late_work_is_fenced():
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    with operation_scope(budget, source_limit=2) as operation:
        admitted = await asyncio.gather(*(
            asyncio.to_thread(budget.admit_source, source(i).identity)
            for i in range(10)
        ))
        assert sum(admitted) == 2
    token = _OPERATION.set(operation)
    try:
        assert not budget.admit_source(source(0).identity)
        assert not budget.admit_source(source(20).identity)
    finally:
        _OPERATION.reset(token)
    assert budget.sources_used == 2


@pytest.mark.asyncio
async def test_initial_search_leaves_novel_source_slots_for_followup():
    budget = TurnBudget(TurnLimits(retrieval_ms=30_000))
    actions = []

    async def execute(action):
        assert budget.reserve_action(action.signature)
        actions.append(action)
        start = 0 if len(actions) == 1 else 45
        rows = [source(i) for i in range(start, start + 45)]
        return {"sources": [s for s in rows if budget.admit_source(s.identity)]}

    result = await acquire_evidence(
        "Which method?", budget=budget, llm=None, execute=execute,
        evidence_views=views, assess=planner, allowed_documents={"doc"},
        iterative=True,
    )
    assert len(actions) == 2
    assert len(result.results[0]["sources"]) == 30
    assert len(result.results[1]["sources"]) == 15
    assert budget.sources_used == 45
    assert result.stop_reason == "source_limit"
    assert budget.planner_calls == 1


@pytest.mark.asyncio
async def test_followup_can_use_remaining_time_beyond_three_seconds():
    now = [0.0]
    budget = TurnBudget(TurnLimits(retrieval_ms=30_000), clock=lambda: now[0])
    allowances = []

    async def execute(action):
        assert budget.reserve_action(action.signature)
        if budget.actions_used == 1:
            now[0] += 6.0
        else:
            allowances.append(_OPERATION.get().remaining_ms())
            # A six-second follow-up fits the turn, but not the old 3s cap.
            now[0] += 6.0
        item = source(budget.actions_used)
        assert budget.admit_source(item.identity)
        return {"sources": [item]}

    result = await acquire_evidence(
        "Which method?", budget=budget, llm=None, execute=execute,
        evidence_views=views, assess=planner, allowed_documents={"doc"},
        iterative=True,
    )
    assert allowances and allowances[0] > 6000
    assert len(result.results) == 2
    assert budget.remaining_ms() >= budget.limits.final_scoring_ms + 1000


@pytest.mark.asyncio
async def test_planner_timeout_keeps_evidence_and_reports_specific_failure():
    budget = TurnBudget(TurnLimits(retrieval_ms=30_000))

    async def execute(action):
        assert budget.reserve_action(action.signature)
        return {"sources": [source(1)]}

    async def timeout(*args, **kwargs):
        raise TimeoutError("private prompt must not become diagnostic text")

    result = await acquire_evidence(
        "Which method?", budget=budget, llm=None, execute=execute,
        evidence_views=views, assess=timeout, allowed_documents={"doc"},
        iterative=True,
    )
    assert result.stop_reason == "planner_timeout"
    assert len(result.results) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("iterative,limits", [
    (False, TurnLimits()), (True, TurnLimits(planner_calls=0)),
    (True, TurnLimits(actions=1)),
])
async def test_no_possible_followup_keeps_full_initial_source_allowance(iterative, limits):
    budget = TurnBudget(limits)

    async def execute(action):
        assert budget.reserve_action(action.signature)
        return {"sources": [
            source(i) for i in range(50) if budget.admit_source(source(i).identity)
        ]}

    result = await acquire_evidence(
        "Which method?", budget=budget, llm=None, execute=execute,
        evidence_views=views, iterative=iterative,
    )
    assert len(result.results[0]["sources"]) == 45
    assert budget.planner_calls == 0
