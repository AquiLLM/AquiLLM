import json

import pytest
from jsonschema import Draft202012Validator

from apps.chat.services.rag_coverage import CoverageAssessment, recheck_support
from apps.chat.tests.test_rag_planner_generation import Completions, action, assess
from lib.retrieval.evidence import SourceEvidence
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.asyncio
async def test_local_planner_lead_is_a_ranking_hint_and_cannot_resolve_an_aspect(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "131072")
    monkeypatch.setenv("RAG_PLANNER_EVIDENCE_LEADS_ENABLED", "1")
    response = {
        "requested_aspects": ["yield"],
        "unresolved_aspects": ["yield"],
        "next_action": action(),
        "evidence_leads": [{"chunk_id": 1, "quote": "Yield measurement methods."}],
    }
    provider = Completions(json.dumps(response))
    assessment = await assess(provider)
    schema = provider.calls[0]["response_format"]["json_schema"]["schema"]
    Draft202012Validator(schema).validate(response)
    assert assessment.next_action.query == "measured yield"
    (lead,) = assessment.leads
    assert (lead.chunk_id, lead.source_fingerprint, lead.start, lead.end) == (
        1,
        "revision-a",
        0,
        26,
    )
    assert not assessment.support and assessment.unresolved_aspects == ("yield",)
    assert not recheck_support(assessment, ()).leads


def test_quote_resolution_accepts_only_a_unique_contiguous_span_in_the_supplied_revision():
    from apps.chat.services.rag_planner_leads import resolve_leads

    text = "⚗ Volume and\n  density are inputs. Alpha Alpha Alpha."
    views = (SourceEvidence(21, "a", 2, "rev-a", text),)
    quote = {"chunk_id": 21, "quote": "Volume and density are inputs."}
    ledger = TurnBudget(TurnLimits())
    (lead,) = resolve_leads([quote], views, ledger)
    assert text[lead.start : lead.end] == "Volume and\n  density are inputs."
    assert resolve_leads([quote, quote], views, ledger) == (lead,)
    for bad in (
        "Alpha Alpha",
        "density was provided.",
        "density are input",
        "  ",
        "X" * 351,
    ):
        assert not resolve_leads([dict(quote, quote=bad)], views, ledger)
    assert not resolve_leads([dict(quote, chunk_id=True)], views, ledger)
    assert not resolve_leads([dict(quote, chunk_id=18)], views, ledger)
    assert not resolve_leads(
        [quote], (*views, SourceEvidence(21, "a", 2, "rev-b", text)), ledger
    )
    exhausted = TurnBudget(TurnLimits(tokenized_codepoints=1))
    assert not resolve_leads([quote], views, exhausted)


def test_lead_binding_does_not_enable_ordinary_planner_support(monkeypatch):
    from dataclasses import replace

    from apps.chat.services.rag_planner_leads import EvidenceLead
    from apps.chat.services.rag_selection_hints import (
        bind_selection_hints,
        consume_selection_hints,
    )
    from apps.chat.tests.test_rag_selection_hint_binding import QUESTION, bound_case
    from apps.documents.services.source_loading import source_runtime_scope

    case = bound_case.__wrapped__(monkeypatch)
    monkeypatch.setenv("RAG_SELECTION_SUPPORT_HINTS_ENABLED", "0")
    monkeypatch.setenv("RAG_PLANNER_EVIDENCE_LEADS_ENABLED", "1")
    ref = case.assessment.support[0]
    lead = EvidenceLead(ref.chunk_id, ref.source_fingerprint, ref.start, ref.end)
    assessment = replace(case.assessment, leads=(lead,))
    with source_runtime_scope(case.runtime):
        assert (
            bind_selection_hints(case.runtime, QUESTION, case.assessment, case.sources)
            is None
        )
        assert bind_selection_hints(case.runtime, QUESTION, assessment, case.sources)
        assert consume_selection_hints(
            case.runtime, case.prepared, case.pool, QUESTION
        )[0][1] == (lead,)
        assert bind_selection_hints(case.runtime, QUESTION, assessment, case.sources)
        monkeypatch.setenv("RAG_PLANNER_EVIDENCE_LEADS_ENABLED", "0")
        assert not consume_selection_hints(
            case.runtime, case.prepared, case.pool, QUESTION
        )
    assert not CoverageAssessment().leads


def test_quoted_lead_keeps_the_original_top_passage_after_a_search_rewrite():
    from apps.chat.services.rag_planner_leads import EvidenceLead
    from apps.chat.services.rag_selection_coordinator import select_prepared_evidence
    from apps.chat.services.rag_selection_types import SelectionLimits, SelectionProfile
    from apps.chat.tests.test_followup_ranking import trial_pool

    pool = trial_pool()
    fourth = pool[3]
    lead = EvidenceLead(fourth.chunk_id, fourth.source_fingerprint, 0, len(fourth.text))
    config = dict(
        mode="adaptive",
        limits=SelectionLimits(1, 1, 50),
        score_status="model",
        question="How is AX-4 storage handled?",
        profile=SelectionProfile("test", 1, 0.1, "v1"),
        support_hints=((fourth, (lead,)),),
        followup_query="AX-4 amber borosilicate curing",
        turn_budget=TurnBudget(TurnLimits()),
    )
    result = select_prepared_evidence(pool, **config)
    assert result.candidates == (pool[0],)
    assert "followup-bm25-rrf" in result.profile.version


def test_evaluation_records_whether_planner_leads_are_enabled(monkeypatch):
    from apps.chat.evals.evidence_effective_config import additional_controls
    from apps.chat.services.rag_planner_leads import LEAD_POLICY

    for flag in (False, True):
        monkeypatch.setenv("RAG_PLANNER_EVIDENCE_LEADS_ENABLED", str(int(flag)))
        assert additional_controls()["planner_evidence_leads"] == {
            "enabled": flag,
            "policy": LEAD_POLICY,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("outside", [False, True])
async def test_a_valid_quote_survives_a_refined_aspect_or_unusable_action(
    monkeypatch, outside
):
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "131072")
    monkeypatch.setenv("RAG_PLANNER_EVIDENCE_LEADS_ENABLED", "1")
    proposed = action("document", document_id="outside" if outside else "doc-a")
    proposed["aspect"] = "temperature adjustment"
    payload = {
        "evidence_leads": [{"chunk_id": 1, "quote": "Yield measurement methods."}],
        "requested_aspects": ["yield"],
        "unresolved_aspects": ["temperature adjustment"],
        "next_action": proposed,
    }
    assessment = await assess(Completions(json.dumps(payload)))
    assert tuple(ref.chunk_id for ref in assessment.leads) == (1,)
    assert not assessment.support
    assert set(assessment.unresolved_aspects) == {"yield", "temperature adjustment"}
    if outside:
        assert assessment.next_action is None
    else:
        assert assessment.next_action.document_id == "doc-a"


@pytest.mark.asyncio
async def test_detailed_unresolved_facets_fit_the_existing_four_aspect_limit(monkeypatch):
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "131072")
    monkeypatch.setenv("RAG_PLANNER_EVIDENCE_LEADS_ENABLED", "1")
    payload = {
        "evidence_leads": [{"chunk_id": 1, "quote": "Yield measurement methods."}],
        "requested_aspects": ["inputs", "procedure", "comparison", "settings"],
        "unresolved_aspects": ["yield", "temperature adjustment", "drying time"],
        "next_action": action(),
    }
    assessment = await assess(Completions(json.dumps(payload)))
    assert assessment.next_action.query == "measured yield"
    assert len(assessment.requested_aspects) == 4
    assert set(payload["unresolved_aspects"]) <= set(assessment.unresolved_aspects)
    assert len(assessment.leads) == 1 and not assessment.support
