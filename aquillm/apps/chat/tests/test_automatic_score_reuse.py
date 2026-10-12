"""No-new-scoring policy still validates real retained acquisition scores."""

from dataclasses import replace
from time import monotonic

import pytest
import requests

from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
from apps.chat.tests.rag_suffix_test_support import fake_pair_transport
from apps.chat.tests.test_rag_selection_scoring import Policy, _authorization, _chunk, _pool
from apps.documents.services import chunk_rerank_selection_provider as factory
from apps.documents.services import chunk_rerank_window_adapter as adapter
from apps.documents.services import rerank_transport_policy
from apps.documents.services.chunk_rerank_pair_capability import invalidate_pair_capability
from apps.documents.services.chunk_rerank_window_acquisition import rerank_windowed
from apps.documents.services.pair_worker_lifecycle import canonical_provider
from lib.retrieval import TurnBudget, TurnLimits


@pytest.fixture
def acquisition(monkeypatch, settings):
    settings.RAG_CACHE_ENABLED = False
    for name, value in {
        "APP_RERANK_PROVIDER": "local",
        "APP_RERANK_BASE_URL": "http://automatic-reuse.test/v1",
        "APP_RERANK_MODEL": "automatic-reuse-test",
        "APP_RERANK_MODEL_REVISION": "b" * 40,
        "RAG_RERANK_TEXT_MODE": "windowed",
        "RAG_RERANK_SHADOW_SCORING_ENABLED": "0",
    }.items():
        monkeypatch.setenv(name, value)
    provider = canonical_provider(deadline=monotonic() + 15)
    budget = TurnBudget(TurnLimits())
    try:
        calls, _ = fake_pair_transport(monkeypatch)
        chunks = tuple(_chunk(i, f"Finding {i}: the measured value is {i} mK.") for i in range(1, 10))
        result = rerank_windowed("measured values?", chunks, 9, turn_budget=budget)
        assert result.score_set.status == "complete"
        assert len(result.score_set.scores) == len(calls) == 9
        assert all(score.window_coverage.is_complete() for score in result.score_set.scores)

        attempts = []

        def forbidden(*args, **kwargs):
            attempts.append(True)
            raise AssertionError("final reuse must not perform external scoring, probing, or score-cache I/O")

        monkeypatch.setattr(requests.sessions.Session, "send", forbidden)
        monkeypatch.setattr(rerank_transport_policy, "discover_rerank_protocol", forbidden)
        monkeypatch.setattr(adapter, "get_window_result", forbidden)
        monkeypatch.setattr(adapter, "set_window_result", forbidden)
        original, resolved = factory.current_selection_scorer, []

        def resolve(**kwargs):
            scorer = original(**kwargs)
            resolved.append(scorer)
            return scorer

        monkeypatch.setattr(factory, "current_selection_scorer", resolve)
        yield chunks, result.score_set, budget, resolved, attempts
    finally:
        budget.close("test_finished")
        invalidate_pair_capability(provider)


@pytest.mark.parametrize("case", ["complete", "runtime_identity", "partial", "query", "source", "revision", "missing"])
def test_automatic_factory_reuses_only_exact_complete_acquisition_without_new_scores(acquisition, monkeypatch, case):
    chunks, scores, budget, resolved, attempts = acquisition
    query = "measured values?"
    if case == "runtime_identity":
        monkeypatch.setenv("APP_RERANK_MODEL_REVISION", "e" * 40)
    elif case == "partial":
        first = scores.scores[0]
        first = replace(first, window_coverage=replace(first.window_coverage, successful_pair_fingerprints=()))
        scores = replace(scores, scores=(first, *scores.scores[1:]))
    elif case == "query":
        query = "different measured values?"
    elif case == "source":
        chunks[0].content += " Corrected value."
    elif case == "revision":
        chunks[0].source_revision = "replacement-storage-revision"
    pool = _pool(chunks, () if case == "missing" else (scores,))
    prepared = prepare_selection_candidates(
        pool=pool, primary_query=query, authorization=_authorization(Policy()),
        deadline=monotonic() + 3, allow_new_scores=False,
        chunk_loader=lambda *_: chunks, turn_budget=budget, source_mode=True,
    )
    assert tuple(c.chunk_id for c in prepared.candidates) == tuple(c.pk for c in chunks)
    assert tuple(c.text for c in prepared.candidates) == tuple(c.content for c in chunks)
    assert prepared.new_pairs == 0
    assert attempts == []
    if case == "complete":
        assert prepared.score_status == "model"
        assert prepared.reused_pairs == 9
        assert prepared.fallback_reason is None
        assert resolved[0].scorer_fingerprint == scores.scorer_fingerprint
    else:
        assert prepared.score_status == "rank_fallback"
        assert prepared.reused_pairs == 0
    assert len(resolved) == (0 if case == "missing" else 1)
    if resolved:
        assert isinstance(resolved[0], adapter.WindowSelectionScorer)
        assert resolved[0].budget is budget
