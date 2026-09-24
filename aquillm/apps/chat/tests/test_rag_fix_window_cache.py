"""Real window envelopes retain identities and opt-in cache behavior."""

from types import SimpleNamespace
from uuid import UUID

from apps.chat.tests.test_rag_fix_cache import backend as backend
from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def test_complete_window_cache_reuses_only_the_exact_score_identity(backend):
    calls = []

    def score(pair, timeout):
        calls.append(pair)
        return 4.0, pair

    def adapter():
        return WindowSelectionScorer(
            SimpleNamespace(score_pair=score),
            budget=TurnBudget(TurnLimits()),
            pair_counter=lambda q, d: len(q) + len(d),
            scorer_identity="cache-test",
            cache_enabled=True,
        )

    row = SimpleNamespace(pk=1, doc_id=UUID(int=1), chunk_number=0, content="evidence")
    first = adapter().score_windows("question", (row,), phase="acquisition")
    assert first.status == "complete"
    assert len(calls) == 1
    assert backend.writes
    second = adapter().score_windows("question", (row,), phase="acquisition")
    assert second.scores == first.scores
    assert len(calls) == 1
    row.content = "new revision"
    third = adapter().score_windows("question", (row,), phase="acquisition")
    assert third.status == "complete"
    assert len(calls) == 2
    assert third.scores[0].source_fingerprint != first.scores[0].source_fingerprint


def test_window_cache_remains_disabled_when_setting_is_off(backend, settings):
    from apps.documents.services.chunk_rerank_score_cache import (
        get_window_result,
        set_window_result,
    )

    settings.RAG_CACHE_ENABLED = False
    budget = TurnBudget(TurnLimits())
    result = SimpleNamespace(status="complete", schema_version="v3-window")
    assert not set_window_result("rrwindow:v1:off", result, budget=budget)
    assert get_window_result("rrwindow:v1:off", budget=budget) is None
    assert backend.contexts == []
