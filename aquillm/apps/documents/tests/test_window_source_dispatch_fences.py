"""Concurrent source attempts retain shared cancellation and real lease fences."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from time import monotonic

import pytest

from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
from apps.documents.tests.rerank_outage_support import chunks
from apps.documents.tests.test_chunk_rerank_window_fences import (
    _verified_window_provider,
)
from apps.documents.tests.test_window_source_dispatch import response
from lib.retrieval import TurnBudget, TurnLimits


def wait_for_release(budget):
    until = monotonic() + 2
    tick = Event()
    while budget._inflight and monotonic() < until:
        tick.wait(0.005)
    assert budget._inflight == 0


@pytest.mark.parametrize("failure", ["provider", "deadline", "close"])
def test_aborted_dispatch_records_live_leases_and_never_publishes_late_scores(
    failure, monkeypatch
):
    monkeypatch.setenv("APP_RERANK_SCORE_CONCURRENCY", "3")
    budget = TurnBudget(TurnLimits())
    provider = _verified_window_provider("http://synthetic/score", budget)
    all_entered, release, lock = Event(), Event(), Lock()
    calls, attempts, writes = [], [], []
    now = [monotonic()]
    adapter = WindowSelectionScorer(
        provider,
        budget=budget,
        pair_counter=provider.verified_pair_counter,
        cache_enabled=True,
        clock=lambda: now[0],
        deadline=now[0] + 1,
    )

    def post(*args, **kwargs):
        text = kwargs["json"]["text_2"]
        with lock:
            calls.append(text)
            if len(calls) == 3:
                all_entered.set()
        assert all_entered.wait(1)
        if text.endswith("1"):
            if failure == "provider":
                return type("Failed", (), {"status_code": 503})()
            if failure == "deadline":
                now[0] += 2
            else:
                budget.close("cancelled")
            return response(provider, 1)
        assert release.wait(2)
        return response(provider, 1)

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post", post
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_window_adapter.get_window_result",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_window_adapter.set_window_result",
        lambda *a, **k: writes.append(a),
    )
    try:
        result = adapter.score_windows(
            "q", chunks(8), phase="acquisition", on_submit=attempts.append
        )
        assert result.status == "unavailable" and not result.scores
        assert len(calls) == sum(attempts) == budget.pairs_used["acquisition"] == 3
        assert budget._inflight == 2, "late real I/O must retain its actual leases"
        assert adapter._scores == {} and writes == []
    finally:
        release.set()
        wait_for_release(budget)
        budget.close("test_complete")
    assert len(calls) == 3 and sum(attempts) == 3
    assert adapter._scores == {} and writes == []


def test_shared_pair_limit_never_turns_queued_sources_into_extra_transports(
    monkeypatch,
):
    monkeypatch.setenv("APP_RERANK_SCORE_CONCURRENCY", "6")
    budget = TurnBudget(TurnLimits(acquisition_pairs=2))
    provider = _verified_window_provider("http://synthetic/score", budget)
    calls, attempts = [], []

    def post(*args, **kwargs):
        calls.append(kwargs["json"])
        return response(provider, 1)

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post", post
    )
    adapter = WindowSelectionScorer(
        provider, budget=budget, pair_counter=provider.verified_pair_counter
    )
    try:
        result = adapter.score_windows(
            "q", chunks(8), phase="acquisition", on_submit=attempts.append
        )
        assert result.status == "unavailable" and not adapter._scores
        assert len(calls) <= sum(attempts) == budget.pairs_used["acquisition"] == 2
    finally:
        wait_for_release(budget)
        budget.close("test_complete")


def test_observer_exception_cannot_leak_real_transport_lease(monkeypatch):
    budget = TurnBudget(TurnLimits())
    provider = _verified_window_provider("http://synthetic/score", budget)
    observed = []

    def observe(count):
        observed.append(count)
        raise RuntimeError("diagnostic receiver unavailable")

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post",
        lambda *a, **k: response(provider, 1),
    )
    adapter = WindowSelectionScorer(
        provider, budget=budget, pair_counter=provider.verified_pair_counter
    )
    try:
        result = adapter.score_windows("q", chunks(3), phase="final", on_submit=observe)
        assert result.status == "complete"
        assert sum(observed) == budget.pairs_used["final"] == 3
        assert budget._inflight == 0
    finally:
        budget.close("test_complete")


def test_unknown_preparation_never_dispatches_other_sources(monkeypatch):
    budget = TurnBudget(TurnLimits())
    provider = _verified_window_provider("http://synthetic/score", budget)
    calls = []
    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post",
        lambda *a, **k: calls.append(k),
    )
    adapter = WindowSelectionScorer(
        provider, budget=budget, pair_counter=lambda q, d: None
    )
    try:
        result = adapter.score_windows("q", chunks(3), phase="acquisition")
        assert result.status == "unavailable" and calls == []
        assert budget.pairs_used["acquisition"] == 0
    finally:
        budget.close("test_complete")


def test_zero_inflight_allowance_returns_rank_fallback_without_executor():
    budget = TurnBudget(TurnLimits(in_flight_pairs=0))
    provider = _verified_window_provider("http://synthetic/score", budget)
    adapter = WindowSelectionScorer(
        provider, budget=budget, pair_counter=provider.verified_pair_counter
    )
    try:
        result = adapter.score_windows("q", chunks(3), phase="acquisition")
        assert result.status == "unavailable" and not adapter._scores
        assert budget.pairs_used["acquisition"] == budget._inflight == 0
    finally:
        budget.close("test_complete")


@pytest.mark.parametrize("cancel", ["budget", "scoring_latch"])
def test_cancelled_caller_returns_before_blocked_real_transports_finish(
    cancel, monkeypatch
):
    monkeypatch.setenv("APP_RERANK_SCORE_CONCURRENCY", "3")
    budget = TurnBudget(TurnLimits())
    provider = _verified_window_provider("http://synthetic/score", budget)
    entered, release, lock = Event(), Event(), Lock()
    calls, attempts = [], []
    adapter = WindowSelectionScorer(
        provider, budget=budget, pair_counter=provider.verified_pair_counter
    )

    def post(*args, **kwargs):
        with lock:
            calls.append(kwargs["json"])
            if len(calls) == 3:
                entered.set()
        assert release.wait(2)
        return response(provider, 1)

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post", post
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                adapter.score_windows,
                "q",
                chunks(8),
                phase="acquisition",
                on_submit=attempts.append,
            )
            try:
                assert entered.wait(1)
                if cancel == "budget":
                    budget.close("cancelled")
                else:
                    adapter.rerank_state.fail()
                result = future.result(timeout=0.3)
                assert result.status == "unavailable"
                assert budget._inflight == 3
                assert sum(attempts) == 3 and not adapter._scores
            finally:
                release.set()
                future.result(timeout=2)
    finally:
        release.set()
        wait_for_release(budget)
        budget.close("test_complete")
    assert len(calls) == 3 and not adapter._scores
