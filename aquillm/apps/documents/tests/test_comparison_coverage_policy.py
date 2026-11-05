"""Default-off controls, literal provenance and final same-turn binding."""

from dataclasses import replace
from contextlib import contextmanager, nullcontext
from hashlib import md5, sha256
from types import SimpleNamespace

import pytest
from django.core.exceptions import ImproperlyConfigured

from apps.chat.services.rag_selection_types import PreparedSelection, SelectionCandidate
from apps.documents.services.comparison_coverage_policy import (
    CoverageGroup, CoverageSource, bind_coverage, comparison_coverage_enabled,
    comparison_policy_identity, selection_coverage_groups,
)
from apps.documents.services.comparison_version_query import parse_version_comparison, resolve_primary_documents
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.retrieval.evidence import PreparedEvidence, SourceEvidence, SourceSpan, fingerprint_prepared_evidence
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

QUESTION = "What are the compatibility changes from the second revision to the third revision of Quartz?"


def test_comparison_opt_in_is_default_off_and_invalid_values_fail_closed(monkeypatch):
    monkeypatch.delenv("RAG_COMPARISON_COVERAGE_ENABLED", raising=False)
    assert comparison_coverage_enabled() is False
    disabled = comparison_policy_identity()
    monkeypatch.setenv("RAG_COMPARISON_COVERAGE_ENABLED", "1")
    assert comparison_coverage_enabled() is True and comparison_policy_identity() != disabled
    monkeypatch.setenv("RAG_COMPARISON_COVERAGE_ENABLED", "experimental")
    with pytest.raises(ImproperlyConfigured, match="RAG_COMPARISON_COVERAGE_ENABLED"):
        comparison_coverage_enabled()


def test_generic_distinct_self_declarations_are_required():
    earlier = SimpleNamespace(chunk_id=1, document_id="left", text="This report presents the second revision of Quartz.")
    later = SimpleNamespace(chunk_id=2, document_id="right", text="This report presents the third revision of Quartz.")
    assert resolve_primary_documents(QUESTION, [earlier, later])["documents"] == ("left", "right")
    for rejected in (
        SimpleNamespace(**{**later.__dict__, "document_id": "left"}),
        SimpleNamespace(**{**later.__dict__, "text": "We will present the third revision of Quartz."}),
        SimpleNamespace(**{**later.__dict__, "text": "This report presents the third revision of Basalt."}),
    ):
        assert resolve_primary_documents(QUESTION, [earlier, rejected])["documents"] == ()
    competing = SimpleNamespace(**{**earlier.__dict__, "document_id": "competing"})
    assert resolve_primary_documents(QUESTION, [earlier, later, competing])["documents"] == ()
    assert parse_version_comparison("What does this project measure?") is None
    assert parse_version_comparison(QUESTION.replace("third", "second")) is None


@pytest.fixture
def selection_binding(monkeypatch):
    from apps.documents.services import comparison_coverage_policy as policy

    monkeypatch.setenv("RAG_COMPARISON_COVERAGE_ENABLED", "1")
    monkeypatch.setattr(policy, "_ranking_controls", lambda: (("full_text", False), ("rrf", False)), raising=False)
    runtime = SourceRuntime(TurnBudget(TurnLimits()), object())
    validated = []
    frozen = SimpleNamespace(representation=object(), dense=SimpleNamespace(manifest_sha256="manifest"),
                             validate=lambda: validated.append(True), database=nullcontext)
    runtime._frozen_retrieval_scope = frozen
    scope = SimpleNamespace(runtime=runtime, representation=frozen.representation, dense=frozen.dense)
    candidates, groups = [], []
    for number, document in ((1, "left"), (2, "right")):
        text = f"Original complete evidence for {document}."
        digest = sha256(text.encode()).hexdigest()
        source = SourceEvidence(number, document, number, digest, text)
        spans = (SourceSpan(number, digest, 0, len(text), text),)
        evidence = PreparedEvidence(source, spans, fingerprint_prepared_evidence(source, spans), 10, "complete")
        candidate = SelectionCandidate(number, document, number, text, 0.4, number, digest, {},
                                       prepared_evidence=evidence, token_cost=10, raw_score=0.4,
                                       scorer_fingerprint="actual-scorer", scoring_kind="pointwise")
        candidates.append(candidate)
        proof = CoverageSource(number, document, number, None, md5(text.encode()).hexdigest(), len(text), digest)
        groups.append(CoverageGroup((proof,), object(), None))
    runtime.comparison_coverage = bind_coverage(scope, QUESTION, groups)
    prepared = PreparedSelection(tuple(candidates), "model", 0, 2, 1.0, None)
    pool = SimpleNamespace(rows=tuple({"chunk_id": c.chunk_id} for c in candidates))
    return runtime, prepared, pool, validated


def test_final_coverage_keeps_exact_scored_objects(selection_binding):
    runtime, prepared, pool, validated = selection_binding
    assert selection_coverage_groups(runtime, prepared, pool, QUESTION) == ()
    with source_runtime_scope(runtime):
        groups = selection_coverage_groups(runtime, prepared, pool, QUESTION)
    assert tuple(group[0] for group in groups) == prepared.candidates
    assert all(group[0] is candidate for group, candidate in zip(groups, prepared.candidates, strict=True))
    assert validated == [True]


def test_unscored_background_candidate_prevents_claiming_complete_model_pool(selection_binding):
    runtime, prepared, pool, _ = selection_binding
    background = replace(prepared.candidates[0], chunk_id=3, doc_id="background", raw_score=None)
    prepared = replace(prepared, candidates=(*prepared.candidates, background))
    pool.rows = (*pool.rows, {"chunk_id": 3})
    with source_runtime_scope(runtime):
        assert selection_coverage_groups(runtime, prepared, pool, QUESTION) == ()


def test_final_revalidation_is_database_bounded_and_owner_checked_after_cleanup(selection_binding):
    runtime, prepared, pool, events = selection_binding

    @contextmanager
    def database():
        events.append("enter")
        yield
        events.append("exit")
        runtime.authorization = object()

    def validate():
        assert events == ["enter"]
        events.append("validate")

    runtime._frozen_retrieval_scope.database = database
    runtime._frozen_retrieval_scope.validate = validate
    with source_runtime_scope(runtime):
        assert selection_coverage_groups(runtime, prepared, pool, QUESTION) == ()
    assert events == ["enter", "validate", "exit"]


@pytest.mark.parametrize("change", ["query", "runtime", "authorization", "policy", "representation", "dense", "partial", "revision", "span", "seal", "expired", "body", "revalidation_owner", "budget", "ranking"])
def test_stale_or_partial_final_coverage_abstains(selection_binding, monkeypatch, change):
    runtime, prepared, pool, validated = selection_binding
    query = QUESTION
    if change == "query":
        query = "Another question?"
    elif change == "runtime":
        original = runtime
        runtime = SourceRuntime(original.budget, original.authorization)
        runtime._frozen_retrieval_scope = original._frozen_retrieval_scope
        runtime.comparison_coverage = original.comparison_coverage
    elif change == "authorization":
        runtime.authorization = object()
    elif change == "policy":
        monkeypatch.setenv("RAG_COMPARISON_COVERAGE_ENABLED", "0")
    elif change == "representation":
        runtime._frozen_retrieval_scope.representation = object()
    elif change == "dense":
        runtime._frozen_retrieval_scope.dense = SimpleNamespace(manifest_sha256="changed")
    elif change == "partial":
        prepared = replace(prepared, candidates=prepared.candidates[:1])
    elif change == "revision":
        prepared = replace(prepared, candidates=(replace(prepared.candidates[0], source_fingerprint="changed"), prepared.candidates[1]))
    elif change == "span":
        first = prepared.candidates[0]
        source = first.prepared_evidence.source
        span = SourceSpan(source.chunk_id, source.source_fingerprint, 0, 8, source.text[:8])
        evidence = PreparedEvidence(source, (span,), fingerprint_prepared_evidence(source, (span,)), 2, "partial")
        prepared = replace(prepared, candidates=(replace(first, prepared_evidence=evidence), prepared.candidates[1]))
    elif change == "expired":
        runtime.budget.close("deadline")
    elif change == "body":
        first = prepared.candidates[0]
        source = replace(first.prepared_evidence.source, text="X" + first.prepared_evidence.source.text[1:])
        span = SourceSpan(source.chunk_id, source.source_fingerprint, 0, len(source.text), source.text)
        evidence = PreparedEvidence(source, (span,), fingerprint_prepared_evidence(source, (span,)), 10, "complete")
        prepared = replace(prepared, candidates=(replace(first, prepared_evidence=evidence), prepared.candidates[1]))
    elif change == "revalidation_owner":
        runtime._frozen_retrieval_scope.validate = lambda: setattr(runtime, "authorization", object())
    elif change == "budget":
        runtime.budget = TurnBudget(TurnLimits())
    elif change == "ranking":
        from apps.documents.services import comparison_coverage_policy as policy
        monkeypatch.setattr(policy, "_ranking_controls", lambda: (("full_text", True), ("rrf", False)))
    else:
        runtime.comparison_coverage = replace(runtime.comparison_coverage, seal=object())
    with source_runtime_scope(runtime):
        assert selection_coverage_groups(runtime, prepared, pool, query) == ()
