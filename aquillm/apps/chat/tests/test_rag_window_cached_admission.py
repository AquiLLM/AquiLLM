"""Cached complete sources remain eligible when no new transport pairs fit."""

import pytest
from django.test import override_settings

from apps.chat.tests.test_rag_window_pair_admission import Scenario
from apps.documents.services.chunk_rerank_score_cache import (
    set_window_result,
    window_cache_key,
)


@pytest.mark.parametrize("cache_kind", ["request", "shared"])
@override_settings(RAG_CACHE_ENABLED=True)
def test_cached_source_is_admitted_without_charging_final_pair_budget(cache_kind):
    case = Scenario(count=2, long_at=1, final_pairs=1)
    cached = case.scorer.score_windows("q", case.chunks[:1], phase="acquisition")
    assert cached.status == "complete"
    assert case.budget.pairs_used["acquisition"] == 2
    if cache_kind == "shared":
        plan = case.scorer.plan("q", case.chunks[0])
        key = window_cache_key(case.scorer.scorer_fingerprint, plan.fingerprint)
        assert set_window_result(key, cached, budget=case.budget)
        case.scorer._scores.clear()
        case.scorer.cache_enabled = True
    case.calls.clear()
    result = case.prepare()
    assert result.score_status == "model"
    assert [candidate.chunk_id for candidate in result.candidates] == [1, 2]
    assert result.new_pairs == case.budget.pairs_used["final"] == 1
    assert case.calls == [("q", "source 2")]


def test_all_cached_sources_remain_model_eligible_with_zero_final_pairs():
    case = Scenario(count=3, long_at=2, final_pairs=0)
    cached = case.scorer.score_windows("q", case.chunks, phase="acquisition")
    assert cached.status == "complete"
    case.calls.clear()
    result = case.prepare()
    assert result.score_status == "model"
    assert len(result.candidates) == 3
    assert result.new_pairs == case.budget.pairs_used["final"] == 0
    assert case.calls == []
