"""Assessment pointers stay turn-local hints and never create source authority."""
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from django.core.exceptions import ImproperlyConfigured

from apps.chat.services.rag_coverage import CoverageAssessment, SupportReference
from apps.chat.services.rag_selection_types import PreparedSelection, SelectionCandidate
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.retrieval.evidence import PreparedEvidence, SourceEvidence, SourceSpan, fingerprint_prepared_evidence
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

QUESTION = "Compare the two approaches, including failure conditions."


@pytest.fixture
def bound_case(monkeypatch):
    monkeypatch.setenv("RAG_SELECTION_SUPPORT_HINTS_ENABLED", "1")
    runtime = SourceRuntime(TurnBudget(TurnLimits()), object())
    frozen = SimpleNamespace(representation=object(), dense=SimpleNamespace(manifest_sha256="dense"))
    runtime._frozen_retrieval_scope = frozen
    sources, candidates = [], []
    for pk, text, score in ((1, "Approach A uses a checkpoint.", .9), (2, "Approach B needs an intact journal.", .85)):
        digest = sha256(text.encode()).hexdigest()
        source = SourceEvidence(pk, f"document-{pk}", pk, digest, text)
        spans = (SourceSpan(pk, digest, 0, len(text), text),)
        evidence = PreparedEvidence(source, spans, fingerprint_prepared_evidence(source, spans), 10, "complete")
        sources.append(source)
        candidates.append(SelectionCandidate(pk, source.document_id, pk, text, score, pk, digest,
            {}, prepared_evidence=evidence, token_cost=10, raw_score=score,
            scorer_fingerprint="original-query-scorer", scoring_kind="pointwise"))
    ref = SupportReference("failure conditions", 2, sources[1].source_fingerprint, 0, len(sources[1].text))
    assessment = CoverageAssessment((ref.aspect,), (ref,), (), None, "assessed")
    prepared = PreparedSelection(tuple(candidates), "model", 0, 2, 0, None)
    pool = SimpleNamespace(rows=tuple({"chunk_id": row.chunk_id} for row in candidates))
    return SimpleNamespace(runtime=runtime, frozen=frozen, sources=tuple(sources), assessment=assessment,
                           prepared=prepared, pool=pool)


@pytest.mark.django_db
def test_hint_consumption_preserves_exact_scored_objects_and_uses_no_database(bound_case, django_assert_num_queries):
    from apps.chat.services.rag_selection_hints import bind_selection_hints, consume_selection_hints
    c = bound_case
    before = (c.runtime.budget.actions_used, c.runtime.budget.sources_used, c.runtime.budget.planner_calls)
    with source_runtime_scope(c.runtime), django_assert_num_queries(0):
        bind_selection_hints(c.runtime, QUESTION, c.assessment, c.sources)
        hints = consume_selection_hints(c.runtime, c.prepared, c.pool, QUESTION)
        assert hints == ((c.prepared.candidates[1], (c.assessment.support[0],)),)
        assert hints[0][0] is c.prepared.candidates[1]
        assert consume_selection_hints(c.runtime, c.prepared, c.pool, QUESTION) == ()
    assert before == (c.runtime.budget.actions_used, c.runtime.budget.sources_used, c.runtime.budget.planner_calls)


def test_hint_gate_defaults_off_and_rejects_invalid_config(bound_case, monkeypatch):
    from apps.chat.services.rag_selection_hints import bind_selection_hints, consume_selection_hints
    c = bound_case
    monkeypatch.delenv("RAG_SELECTION_SUPPORT_HINTS_ENABLED")
    with source_runtime_scope(c.runtime):
        bind_selection_hints(c.runtime, QUESTION, c.assessment, c.sources)
        assert consume_selection_hints(c.runtime, c.prepared, c.pool, QUESTION) == ()
        monkeypatch.setenv("RAG_SELECTION_SUPPORT_HINTS_ENABLED", "maybe")
        with pytest.raises(ImproperlyConfigured):
            bind_selection_hints(c.runtime, QUESTION, c.assessment, c.sources)


@pytest.mark.parametrize("change", ["query", "authorization", "budget", "runtime", "representation", "dense", "flag", "source", "document", "partial_span", "missing_source", "partial_scores", "ranking", "expired"])
def test_changed_binding_or_missing_counterpart_disables_the_whole_hint_set(bound_case, monkeypatch, change):
    from apps.chat.services.rag_selection_hints import bind_selection_hints, consume_selection_hints
    c = bound_case
    query, runtime, prepared = QUESTION, c.runtime, c.prepared
    with source_runtime_scope(runtime):
        bind_selection_hints(runtime, query, c.assessment, c.sources)
    if change == "query": query = "Another question?"
    elif change == "authorization": runtime.authorization = object()
    elif change == "budget": runtime.budget = TurnBudget(TurnLimits())
    elif change == "runtime":
        clone = SourceRuntime(runtime.budget, runtime.authorization)
        clone._frozen_retrieval_scope = runtime._frozen_retrieval_scope
        clone._selection_support_hints = runtime._selection_support_hints
        runtime = clone
    elif change == "representation": c.frozen.representation = object()
    elif change == "dense": c.frozen.dense.manifest_sha256 = "changed"
    elif change == "flag": monkeypatch.setenv("RAG_SELECTION_SUPPORT_HINTS_ENABLED", "0")
    elif change in ("source", "document", "partial_span"):
        row = prepared.candidates[1]
        if change == "source": row = replace(row, source_fingerprint="changed")
        elif change == "document": row = replace(row, doc_id="different-document")
        else:
            source = row.prepared_evidence.source
            span = SourceSpan(source.chunk_id, source.source_fingerprint, 0, 8, source.text[:8])
            evidence = PreparedEvidence(source, (span,), fingerprint_prepared_evidence(source, (span,)), 2, "partial")
            row = replace(row, prepared_evidence=evidence)
        prepared = replace(prepared, candidates=(prepared.candidates[0], row))
    elif change == "missing_source": prepared = replace(prepared, candidates=prepared.candidates[:1])
    elif change == "partial_scores": prepared = replace(prepared, score_status="partial_model_ranked")
    elif change == "ranking":
        from apps.documents.services import chunk_search_rank_policy
        monkeypatch.setattr(chunk_search_rank_policy, "controls", lambda: {"full_text": True})
    else: runtime.budget.close("deadline")
    with source_runtime_scope(runtime):
        assert consume_selection_hints(runtime, prepared, c.pool, query) == ()


def test_out_of_range_or_invented_support_cannot_be_bound(bound_case):
    from apps.chat.services.rag_selection_hints import bind_selection_hints, consume_selection_hints
    c = bound_case
    for ref in (replace(c.assessment.support[0], end=9999), replace(c.assessment.support[0], source_fingerprint="invented")):
        assessment = replace(c.assessment, support=(ref,))
        with source_runtime_scope(c.runtime):
            bind_selection_hints(c.runtime, QUESTION, assessment, c.sources)
            assert consume_selection_hints(c.runtime, c.prepared, c.pool, QUESTION) == ()


@pytest.mark.parametrize("before_bind", [False, True])
def test_same_turn_object_cannot_exchange_its_owner_budget(bound_case, before_bind):
    from apps.chat.services.rag_selection_hints import bind_selection_hints, consume_selection_hints
    from lib.llm.turn_context import TurnContext, bind_turn

    c = bound_case
    state = TurnContext(c.runtime.budget)
    with bind_turn(state), source_runtime_scope(c.runtime):
        if before_bind:
            state.budget = TurnBudget(TurnLimits())
        bind_selection_hints(c.runtime, QUESTION, c.assessment, c.sources)
        if not before_bind:
            state.budget = TurnBudget(TurnLimits())
        assert consume_selection_hints(c.runtime, c.prepared, c.pool, QUESTION) == ()


def test_reentrant_capture_and_abandoned_handoff_cannot_reuse_hints(bound_case):
    from apps.chat.services.rag_selection_hints import (
        bind_selection_hints, clear_selection_hints, consume_selection_hints,
    )

    c = bound_case
    with source_runtime_scope(c.runtime):
        original = bind_selection_hints(c.runtime, QUESTION, c.assessment, c.sources)
        assert original is not None
        assert bind_selection_hints(c.runtime, QUESTION, c.assessment, c.sources) is None
        assert consume_selection_hints(c.runtime, c.prepared, c.pool, QUESTION) == ()
        abandoned = bind_selection_hints(c.runtime, QUESTION, c.assessment, c.sources)
        clear_selection_hints(c.runtime, abandoned)
        assert consume_selection_hints(c.runtime, c.prepared, c.pool, QUESTION) == ()
        replacement = bind_selection_hints(c.runtime, QUESTION, c.assessment, c.sources)
        clear_selection_hints(c.runtime, original)
        assert c.runtime._selection_support_hints is replacement
        assert consume_selection_hints(c.runtime, c.prepared, c.pool, QUESTION)
