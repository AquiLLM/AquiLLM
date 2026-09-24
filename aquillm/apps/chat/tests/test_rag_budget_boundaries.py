"""Concurrent inference diagnostics and finite fail-open cache work."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def test_parallel_operations_report_only_their_own_attempts():
    from apps.documents.services.chunk_rerank_window_scores import score_window_plan

    gate = Barrier(2)

    class Scorer:
        def score_pair(self, pair, timeout):
            gate.wait(2)
            return 1.0, pair

    plan = SimpleNamespace(
        preparation_coverage="complete",
        query="q",
        required_window_ids=("one",),
        windows=(SimpleNamespace(text="evidence"),),
    )
    budget = TurnBudget(TurnLimits())
    with ThreadPoolExecutor(2) as executor:
        results = list(
            executor.map(
                lambda _: score_window_plan(plan, scorer=Scorer(), budget=budget),
                range(2),
            )
        )
    assert budget.pairs_used["acquisition"] == 2
    assert [r.attempted_pairs for r in results] == [1, 1]


def test_preservation_cache_has_finite_transport_and_reserves_completion(
    settings, monkeypatch
):
    from apps.documents.services import bounded_rag_cache
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_runtime_scope,
    )

    captured = []
    closed = []

    class Backend:
        def __init__(self, location, params):
            captured.append(params)
            self._cache = SimpleNamespace(
                _pools={0: SimpleNamespace(disconnect=lambda: closed.append(True))}
            )

        def get(self, key):
            return "cached"

    monkeypatch.setattr(bounded_rag_cache, "RedisCache", Backend)
    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": "redis://local/0",
            "OPTIONS": {},
        }
    }
    budget = TurnBudget(TurnLimits())
    with source_runtime_scope(SourceRuntime(budget, None)):
        assert bounded_rag_cache.cache_operation("get", "safe") == "cached"
        budget.close("cancelled")
        assert bounded_rag_cache.cache_operation("get", "safe") is None
    assert captured[0]["OPTIONS"]["socket_connect_timeout"] <= 0.2
    assert captured[0]["OPTIONS"]["socket_timeout"] <= 0.2
    assert closed == [True]
