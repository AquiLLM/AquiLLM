"""Whole cache waits and reusable publication remain bounded outside ledger locks."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.fixture
def backend(settings, monkeypatch):
    from apps.documents.services import bounded_rag_cache

    bounded_rag_cache._pointers.clear()
    bounded_rag_cache._warm.clear()
    settings.RAG_CACHE_ENABLED = True

    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": "redis://fixture/0",
            "OPTIONS": {},
        }
    }
    state = SimpleNamespace(
        values={}, writes=[], gate=None, phase=None, entered=Event(), contexts=[]
    )

    def block(phase):
        if state.phase == phase:
            state.entered.set()
            assert state.gate.wait(3)

    class Backend:
        def __init__(self, *args):
            from apps.documents.services.source_loading import current_source_runtime

            state.contexts.append(current_source_runtime())
            block("dns")
            self._cache = SimpleNamespace(
                _pools={0: SimpleNamespace(disconnect=lambda: block("cleanup"))}
            )

        def set(self, key, value, timeout):
            block("transport")
            state.values[key] = value
            state.writes.append((key, timeout))
            return True

        def get(self, key):
            block("read")
            return state.values.get(key)

    monkeypatch.setattr(bounded_rag_cache, "RedisCache", Backend)
    return state


def invoke(budget, operation, key, *args, **kwargs):
    from apps.documents.services.bounded_rag_cache import cache_operation

    with source_runtime_scope(SourceRuntime(budget, None)):
        return cache_operation(operation, key, *args, **kwargs)


@pytest.mark.parametrize("phase", ["dns", "transport", "cleanup"])
def test_cache_wait_and_cancel_are_bounded_and_late_stage_is_unpublished(
    backend, phase
):
    backend.phase, backend.gate = phase, Event()
    budget = TurnBudget(TurnLimits())
    closed = Event()
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(
            invoke, budget, "set", "blocked-" + phase, "value", timeout=30
        )
        assert backend.entered.wait(1)
        closer = Thread(
            target=lambda: (budget.close("cancelled"), closed.set()), daemon=True
        )
        closer.start()
        try:
            assert closed.wait(0.1), (
                "network operation held cancellation-critical ledger lock"
            )
            assert pending.result(timeout=0.6) is None
        finally:
            backend.gate.set()
            closer.join(1)
        assert invoke(TurnBudget(TurnLimits()), "get", "blocked-" + phase) is None
    assert all(context is None for context in backend.contexts)
    assert all(
        key.startswith("preservation-stage:") and 0 < ttl <= 30
        for key, ttl in backend.writes
    )


def test_successful_same_process_reuse_then_delete_and_expiry(backend, monkeypatch):
    from apps.documents.services import bounded_rag_cache

    first, second = TurnBudget(TurnLimits()), TurnBudget(TurnLimits())
    assert invoke(first, "set", "reuse", {"score": 7}, timeout=30)
    first.close("finished")
    assert invoke(second, "get", "reuse") == {"score": 7}
    assert invoke(second, "delete", "reuse")
    assert invoke(second, "get", "reuse") is None
    assert invoke(second, "set", "expire", "old", timeout=1)
    monkeypatch.setattr(bounded_rag_cache, "monotonic", lambda: float("inf"))
    assert invoke(second, "get", "expire") is None


@pytest.mark.parametrize("phase", ["dns", "transport"])
def test_window_cache_with_only_explicit_budget_has_no_blocking_outer_fence(
    backend, monkeypatch, phase
):
    from apps.documents.services import (
        chunk_rerank_score_cache,
        chunk_rerank_score_transport,
    )

    monkeypatch.setattr(
        chunk_rerank_score_transport, "serialize_score_set", lambda _: {"valid": True}
    )
    backend.phase, backend.gate = phase, Event()
    budget = TurnBudget(TurnLimits())
    result = SimpleNamespace(schema_version="v3-window", status="complete")
    closed = Event()
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(
            chunk_rerank_score_cache.set_window_result,
            "rrwindow:v1:test",
            result,
            budget=budget,
        )
        # The configured Redis adapter must be reached even without SourceRuntime.
        try:
            assert backend.entered.wait(1)
            closer = Thread(
                target=lambda: (budget.close("cancelled"), closed.set()), daemon=True
            )
            closer.start()
            assert closed.wait(0.1)
            assert not pending.result(timeout=0.6)
        finally:
            backend.gate.set()
        assert invoke(TurnBudget(TurnLimits()), "get", "rrwindow:v1:test") is None


def test_worker_population_is_bounded_without_a_queue(monkeypatch):
    from threading import BoundedSemaphore

    from apps.documents.services import rag_cache_worker

    monkeypatch.setattr(rag_cache_worker, "_slots", BoundedSemaphore(4))
    release, finished = Event(), [Event() for _ in range(4)]
    calls = []

    def blocked(index):
        calls.append(index)
        release.wait(2)
        finished[index].set()

    try:
        for index in range(4):
            assert rag_cache_worker.bounded_cache_job(
                lambda index=index: blocked(index),
                TurnBudget(TurnLimits()),
                seconds=0.02,
            ) == (False, None)
        assert rag_cache_worker.bounded_cache_job(
            lambda: calls.append(99), TurnBudget(TurnLimits()), seconds=0.02
        ) == (False, None)
        assert sorted(calls) == list(range(4))
    finally:
        release.set()
        assert all(event.wait(1) for event in finished)


@pytest.mark.parametrize("stop", ["expiry", "operation", "timeout"])
def test_late_cache_result_never_reaches_a_live_reader(backend, stop):
    from lib.retrieval.operation import operation_scope

    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    assert invoke(budget, "set", "read-fence", "available", timeout=30)
    backend.phase, backend.gate = "read", Event()
    handle = []

    def get():
        with operation_scope(budget) as operation:
            handle.append(operation)
            return invoke(budget, "get", "read-fence")

    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(get)
        assert backend.entered.wait(1)
        if stop == "expiry":
            now[0] = 16.0
        elif stop == "operation":
            with budget._lock:
                handle[0].live = False
        try:
            assert pending.result(timeout=0.6) is None
        finally:
            backend.gate.set()


def test_pointer_capacity_and_warm_legacy_priming_are_bounded(backend, monkeypatch):
    from apps.documents.services import bounded_rag_cache, rag_cache

    monkeypatch.setattr(bounded_rag_cache, "MAX_POINTERS", 2)
    monkeypatch.setattr(bounded_rag_cache, "MAX_WARM_POINTERS", 1)
    budget = TurnBudget(TurnLimits())
    for key in ["one", "two", "three"]:
        assert invoke(budget, "set", key, key, timeout=30)
    assert len(bounded_rag_cache._pointers) == 2
    assert invoke(budget, "get", "one") is None
    key = rag_cache.rerank_capability_cache_key("http://fixture/v1", "model")
    capability = {"endpoint": "http://fixture/score", "shape": "score_single_text_pair"}
    backend.values[key] = capability  # Explicit warm bootstrap uses the legacy key.
    monkeypatch.setattr(rag_cache, "_rag_enabled", lambda: True)
    from time import monotonic

    from apps.documents.services.chunk_rerank_selection_provider import (
        current_selection_scorer,
    )

    monkeypatch.setenv("APP_RERANK_PROVIDER", "local")
    monkeypatch.setenv("APP_RERANK_BASE_URL", "http://fixture/v1")
    monkeypatch.setenv("APP_RERANK_MODEL", "model")
    assert (
        current_selection_scorer(
            deadline=monotonic() + 1, turn_budget=budget, windowed=False
        )
        is not None
    )
    assert (
        rag_cache.get_cached_rerank_capability(
            "http://fixture/v1", "model", budget=budget
        )
        == capability
    )
    assert invoke(budget, "delete", key)
    assert (
        invoke(budget, "get", key) is None
    )  # Cannot revive the still-present old record.
    backend.values["rrcap:another"] = capability
    assert invoke(budget, "get", "rrcap:another") is None
    assert len(bounded_rag_cache._warm) == 1
