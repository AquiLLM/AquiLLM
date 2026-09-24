"""Scripted acquisition behavior without provider/network inference."""

from types import SimpleNamespace

import pytest

from lib.retrieval.evidence import SourceEvidence, fingerprint_source
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def source(pk, text):
    return SourceEvidence(pk, "doc", pk, fingerprint_source(text), text)


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [1, 2, 3])
async def test_targeted_actions_stop_at_cap_without_third_planner(cap):
    from apps.chat.services.rag_acquisition import acquire_evidence
    from apps.chat.services.rag_coverage import AcquisitionAction, CoverageAssessment

    budget = TurnBudget(TurnLimits(actions=cap))
    seen, planned = [], []

    async def execute(action):
        seen.append(action)
        assert budget.reserve_action(action.signature)
        return {"sources": [source(len(seen), f"method {len(seen)}")]}

    async def assess(question, views, anchors, **kwargs):
        planned.append(views)
        return CoverageAssessment(
            ("yield",),
            (),
            ("yield",),
            AcquisitionAction("vector", f"yield {len(planned)}", aspect="yield"),
            "unknown",
        )

    result = await acquire_evidence(
        "What was the measured yield?",
        budget=budget,
        llm=None,
        execute=execute,
        evidence_views=lambda results: tuple(s for r in results for s in r["sources"]),
        assess=assess,
        anchors=SimpleNamespace(unresolved_references=()),
        allowed_documents={"doc"},
        iterative=True,
    )
    assert len(seen) == cap
    assert len(planned) == cap - 1
    assert result.rounds == cap
    assert result.assessment.certainty != "complete"


@pytest.mark.asyncio
async def test_no_new_revision_stops_expansion():
    from apps.chat.services.rag_acquisition import acquire_evidence
    from apps.chat.services.rag_coverage import AcquisitionAction, CoverageAssessment

    budget = TurnBudget(TurnLimits())
    calls = []

    async def execute(action):
        calls.append(action)
        assert budget.reserve_action(action.signature)
        return {"sources": [source(1, "method only")]}

    async def assess(*args, **kwargs):
        return CoverageAssessment(
            ("yield",),
            (),
            ("yield",),
            AcquisitionAction("vector", "yield", aspect="yield"),
            "unknown",
        )

    result = await acquire_evidence(
        "What was the measured yield?",
        budget=budget,
        llm=None,
        execute=execute,
        evidence_views=lambda results: tuple(s for r in results for s in r["sources"]),
        assess=assess,
        allowed_documents={"doc"},
        iterative=True,
    )
    assert len(calls) == 2
    assert result.stop_reason == "no_progress"


@pytest.mark.asyncio
async def test_narrow_supported_lookup_has_no_planner():
    from apps.chat.services.rag_acquisition import acquire_evidence

    budget = TurnBudget(TurnLimits())

    async def execute(action):
        assert budget.reserve_action(action.signature)
        return {"sources": [source(1, "The measured yield was 42 percent.")]}

    async def forbidden(*args, **kwargs):
        pytest.fail("adequate narrow lookup must avoid planner")

    result = await acquire_evidence(
        "What was the measured yield?",
        budget=budget,
        llm=None,
        execute=execute,
        evidence_views=lambda results: tuple(s for r in results for s in r["sources"]),
        assess=forbidden,
        allowed_documents={"doc"},
        iterative=True,
    )
    assert result.rounds == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["malformed", "timeout", "duplicate", "outside", "empty_recovery"]
)
async def test_planner_failures_preserve_available_sources(failure):
    import json

    from apps.chat.services.rag_acquisition import acquire_evidence
    from lib.llm.types.response import LLMResponse

    budget = TurnBudget(TurnLimits())
    question = "What was the measured yield?"
    calls = []

    async def execute(action):
        calls.append(action)
        assert budget.reserve_action(action.signature)
        return {
            "sources": []
            if failure == "empty_recovery" and len(calls) == 1
            else [
                source(
                    len(calls),
                    "The measured yield was 42 percent."
                    if len(calls) > 1
                    else "method only",
                )
            ]
        }

    async def message(**kwargs):
        assert kwargs["max_tokens"] == 512
        if failure == "timeout":
            raise TimeoutError
        action = {
            "kind": "vector",
            "query": question.upper() if failure == "duplicate" else "measured yield",
            "aspect": "yield",
        }
        if failure == "outside":
            action = {
                "kind": "document",
                "query": "yield",
                "document_id": "outside",
                "aspect": "yield",
            }
        payload = json.dumps(
            {
                "requested_aspects": ["yield"],
                "support": [],
                "unresolved_aspects": ["yield"],
                "next_action": action,
            }
        )
        return LLMResponse(
            text="not JSON" if failure == "malformed" else payload,
            tool_call={},
            stop_reason="stop",
            input_usage=0,
            output_usage=0,
            model="fake",
        )

    acquired = await acquire_evidence(
        question,
        budget=budget,
        llm=SimpleNamespace(get_message=message, base_args={}),
        execute=execute,
        evidence_views=lambda results: tuple(s for r in results for s in r["sources"]),
        allowed_documents={"doc"},
        iterative=True,
    )
    assert len(calls) == (2 if failure == "empty_recovery" else 1)
    assert budget.planner_calls == 1
    assert any(r["sources"] for r in acquired.results)


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes_original_budget():
    import asyncio

    from apps.chat.services.rag_acquisition import acquire_evidence

    budget = TurnBudget(TurnLimits())

    async def execute(action):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await acquire_evidence(
            "yield",
            budget=budget,
            llm=None,
            execute=execute,
            evidence_views=lambda _: (),
            iterative=True,
        )
    assert not budget.can_publish()
