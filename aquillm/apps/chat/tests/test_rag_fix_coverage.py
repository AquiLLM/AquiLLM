"""Broad narrative needs assessment; exact measurements retain zero-call proofs."""

from types import SimpleNamespace
import json

import pytest

from apps.chat.services.rag_acquisition import acquire_evidence
from apps.chat.services.rag_coverage import needs_coverage_assessment, recheck_support
from lib.retrieval.evidence import SourceEvidence, SourceSpan
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.parametrize(
    "field,value",
    [
        ("release codename", "Borealis"),
        ("support owner", "Platform Operations"),
        ("policy status", "Approved"),
        ("effective date", "September 22, 2026"),
        ("measured yield", "42 percent"),
    ],
)
@pytest.mark.asyncio
async def test_narrative_planner_keeps_exact_support_after_selection(field, value, monkeypatch):
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    question = f"What is the {field}?"
    text = f"Background. The {field} is {value}."
    source = SourceEvidence(1, "doc", 0, "revision", text)
    assert needs_coverage_assessment(question, (source,), None)
    budget = TurnBudget(TurnLimits())

    async def execute(action):
        budget.reserve_action(action.signature)
        return {"sources": (source,)}

    calls = []
    async def planner(**kwargs):
        calls.append(kwargs)
        supplied = json.loads(kwargs["messages"][0]["content"])
        assert supplied["question"] == question
        assert supplied["evidence"][0]["text"] == text
        assert kwargs["max_tokens"] == 512 and budget.planner_calls == 1
        return SimpleNamespace(text=json.dumps({"requested_aspects": [field],
            "support": [{"aspect": field, "chunk_id": source.chunk_id,
                "source_fingerprint": source.source_fingerprint, "start": 12, "end": len(text)}],
            "unresolved_aspects": [], "next_action": None}))

    acquired = await acquire_evidence(
        question,
        budget=budget,
        llm=SimpleNamespace(base_args={}, get_message=planner),
        execute=execute,
        iterative=True,
        evidence_views=lambda results: tuple(s for r in results for s in r["sources"]),
    )
    assert len(calls) == budget.planner_calls == 1
    assert acquired.rounds == budget.actions_used == 1
    assert acquired.assessment.requested_aspects == (field,)
    assert acquired.assessment.support
    for delivered in [
        (),
        (SimpleNamespace(spans=(SourceSpan(1, "revision", 0, 11, "Background."),)),),
    ]:
        checked = recheck_support(acquired.assessment, delivered)
        assert checked.unresolved_aspects == (field,)
        assert not checked.support
    full = (SimpleNamespace(spans=(SourceSpan(1, "revision", 0, len(text), text),)),)
    assert recheck_support(acquired.assessment, full).support


@pytest.mark.parametrize(
    "text",
    [
        "The policy status is Approved except for archived releases.",
        "The release codename is Borealis. The release codename is Aurora.",
        "The release codename is unknown.",
    ],
)
def test_qualified_conflicting_or_unknown_nonnumeric_fields_require_assessment(text):
    field = "policy status" if "policy" in text else "release codename"
    assert needs_coverage_assessment(
        f"What is the {field}?", (SourceEvidence(1, "doc", 0, "r", text),), None
    )


@pytest.mark.asyncio
async def test_exact_measurement_needs_no_planner_and_keeps_full_span():
    question, text = "What is the measured yield?", "The measured yield is 42 percent."
    source = SourceEvidence(1, "doc", 0, "revision", text)
    assert not needs_coverage_assessment(question, (source,), None)
    budget = TurnBudget(TurnLimits())
    async def execute(action):
        assert budget.reserve_action(action.signature)
        return {"sources": (source,)}
    async def forbidden(*args, **kwargs):
        pytest.fail("closed exact measurement must not dispatch planner")
    acquired = await acquire_evidence(question, budget=budget, llm=None, execute=execute,
        iterative=True, evidence_views=lambda results: tuple(s for r in results for s in r["sources"]),
        assess=forbidden)
    assert budget.planner_calls == 0 and budget.actions_used == 1
    assert acquired.assessment.support[0].start == 0
    assert acquired.assessment.support[0].end == len(text)
    assert not recheck_support(acquired.assessment, ()).support
    delivered = (SimpleNamespace(spans=(SourceSpan(1, "revision", 0, len(text), text),)),)
    assert recheck_support(acquired.assessment, delivered).support
