"""Routine numeric and nonnumeric lookups retain exact final-span obligations."""

from types import SimpleNamespace

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
async def test_zero_planner_lookup_keeps_support_after_selection(field, value):
    question = f"What is the {field}?"
    text = f"Background. The {field} is {value}."
    source = SourceEvidence(1, "doc", 0, "revision", text)
    assert not needs_coverage_assessment(question, (source,), None)
    budget = TurnBudget(TurnLimits())

    async def execute(action):
        budget.reserve_action(action.signature)
        return {"sources": (source,)}

    async def planner(*args, **kwargs):
        pytest.fail("unambiguous exact lookup must not need a planner")

    acquired = await acquire_evidence(
        question,
        budget=budget,
        llm=None,
        execute=execute,
        iterative=True,
        evidence_views=lambda results: tuple(s for r in results for s in r["sources"]),
        assess=planner,
    )
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
