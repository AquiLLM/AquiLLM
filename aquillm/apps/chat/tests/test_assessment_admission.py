"""Assess existing evidence independently of permission to issue another search."""

import asyncio
from types import SimpleNamespace

import pytest

from apps.chat.services.rag_acquisition import acquire_evidence
from apps.chat.services.rag_coverage import AcquisitionAction, CoverageAssessment, SupportReference
from apps.chat.tests.test_initial_acquisition_deadline import clock  # real asyncio/ledger clock fixture
from lib.retrieval.evidence import SourceEvidence, fingerprint_source
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

QUESTION = "Explain the mechanism and its failure condition."


def case(now, *, remaining=10_000, action_limit=3, source_limit=45, empty=False):
    budget = TurnBudget(TurnLimits(retrieval_ms=30_000, actions=action_limit,
        unique_sources=source_limit), clock=lambda: now[0])
    text = "The mechanism uses a journal. Its failure condition is unresolved."
    source = SourceEvidence(1, "doc", 1, fingerprint_source(text), text)
    assert budget.reserve_action("original-acquisition")
    assert budget.admit_source(source.identity)
    now[0] = (budget.limits.retrieval_ms - remaining) / 1000
    ref = SupportReference("mechanism", 1, source.source_fingerprint, 0, 29)
    return SimpleNamespace(budget=budget, source=source, reference=ref,
        initial=({"sources": [] if empty else [source]},))


def assessment(c, *, followup=False):
    return CoverageAssessment(("mechanism", "failure condition"), (c.reference,),
        ("failure condition",), AcquisitionAction("vector", "Find failure conditions",
        aspect="failure condition") if followup else None, "assessed")


async def acquire(c, assess):
    async def forbidden(action):
        pytest.fail("no additional search is allowed in this case")

    return await acquire_evidence(QUESTION, budget=c.budget, llm=None,
        execute=forbidden, evidence_views=lambda results: tuple(s for r in results for s in r["sources"]),
        initial_results=c.initial, initial_acquired=True, assess=assess,
        allowed_documents={"doc"}, iterative=True)


@pytest.mark.asyncio
async def test_ten_seconds_admits_assessment_without_reserving_a_future_search(clock):
    now, elapse = clock
    c = case(now)

    async def assess(*args, **kwargs):
        assert kwargs["budget"] is c.budget
        await elapse(1)
        return assessment(c)

    result = await acquire(c, assess)
    assert c.budget.planner_calls == 1
    assert result.assessment.support == (c.reference,)
    assert result.stop_reason == "assessed"
    assert (c.budget.actions_used, c.budget.sources_used, c.budget.remaining_ms()) == (1, 1, 9000)


@pytest.mark.asyncio
async def test_assessment_keeps_references_when_remaining_action_time_is_insufficient(clock):
    now, elapse = clock
    c = case(now)

    async def assess(*args, **kwargs):
        await elapse(4)
        return assessment(c, followup=True)

    result = await acquire(c, assess)
    assert c.budget.planner_calls == 1
    assert result.assessment.support == (c.reference,)
    assert result.stop_reason == "completion_reserve"
    assert (c.budget.actions_used, c.budget.sources_used, c.budget.remaining_ms()) == (1, 1, 6000)


@pytest.mark.asyncio
async def test_assessment_still_preserves_its_own_completion_and_final_scoring_time(clock):
    now, _ = clock
    c = case(now, remaining=8000)

    async def forbidden(*args, **kwargs):
        pytest.fail("planner needs its 5s allowance plus the existing 4s completion reserve")

    result = await acquire(c, forbidden)
    assert result.stop_reason == "completion_reserve"
    assert c.budget.planner_calls == 0
    assert c.budget.remaining_ms() == 8000


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", ["action", "source"])
@pytest.mark.parametrize("followup", [False, True])
async def test_existing_evidence_can_be_assessed_at_search_capacity_without_exceeding_it(clock, cap, followup):
    now, _ = clock
    c = case(now, remaining=15_000, action_limit=1 if cap == "action" else 3,
        source_limit=1 if cap == "source" else 45)

    async def assess(*args, **kwargs):
        return assessment(c, followup=followup)

    result = await acquire(c, assess)
    assert c.budget.planner_calls == 1
    assert result.assessment.support == (c.reference,)
    assert result.stop_reason == (cap + "_limit" if followup else "assessed")
    assert (c.budget.actions_used, c.budget.sources_used) == (1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", ["action", "source"])
async def test_empty_evidence_without_search_capacity_does_not_call_a_model(clock, cap):
    now, _ = clock
    c = case(now, remaining=15_000, action_limit=1 if cap == "action" else 3,
        source_limit=1 if cap == "source" else 45, empty=True)

    async def forbidden(*args, **kwargs):
        pytest.fail("there is neither evidence to assess nor capacity to acquire any")

    result = await acquire(c, forbidden)
    assert c.budget.planner_calls == 0
    assert result.stop_reason == cap + "_limit"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "cancelled"])
async def test_newly_admitted_assessment_cannot_dispatch_after_timeout_or_cancellation(clock, failure):
    now, elapse = clock
    c = case(now)

    async def assess(*args, **kwargs):
        if failure == "cancelled":
            await elapse(1)
            raise asyncio.CancelledError
        await elapse(6)
        pytest.fail("late planner result must be cancelled at its existing 5s cap")

    if failure == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await acquire(c, assess)
        assert not c.budget.can_publish()
    else:
        result = await acquire(c, assess)
        assert result.stop_reason == "planner_timeout"
        assert result.results == c.initial
        assert c.budget.remaining_ms() >= 4000
    assert (c.budget.planner_calls, c.budget.actions_used, c.budget.sources_used) == (1, 1, 1)
