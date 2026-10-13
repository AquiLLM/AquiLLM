"""Cross-source dispatch keeps exact scores while bounding real leased I/O."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
from apps.documents.tests.rerank_outage_support import chunks
from apps.documents.tests.test_chunk_rerank_window_fences import (
    _verified_window_provider,
)
from lib.retrieval import TurnBudget, TurnLimits


def response(provider, value):
    return SimpleNamespace(
        status_code=200,
        url=provider.endpoint,
        json=lambda: {
            "model": "synthetic",
            "score": value,
            "usage": {"prompt_tokens": 7},
        },
    )


@pytest.mark.parametrize("phase", ["acquisition", "final"])
@pytest.mark.parametrize("configured, cap, width", [(1, 6, 1), (3, 6, 3), (16, 4, 4)])
def test_real_provider_leases_overlap_only_to_configured_bound(
    phase, configured, cap, width, monkeypatch
):
    monkeypatch.setenv("APP_RERANK_SCORE_CONCURRENCY", str(configured))
    budget = TurnBudget(TurnLimits(in_flight_pairs=cap))
    provider = _verified_window_provider("http://synthetic/score", budget)
    adapter = WindowSelectionScorer(
        provider, budget=budget, pair_counter=provider.verified_pair_counter
    )
    rows = chunks(8)
    entered, release, lock = Event(), Event(), Lock()
    context = ContextVar("dispatch_context", default=None)
    token = context.set("captured-turn")
    active, maximum, calls = [0], [0], []

    def post(*args, **kwargs):
        assert context.get() == "captured-turn"
        pair = kwargs["json"]["text_1"], kwargs["json"]["text_2"]
        with lock:
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
            calls.append(pair)
            assert budget._inflight <= width
            if active[0] == width:
                entered.set()
        try:
            assert release.wait(2)
            return response(provider, float(pair[1].split()[-1]))
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(
        "apps.documents.services.chunk_rerank_selection_provider.requests.post", post
    )
    attempts = []
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                copy_context().run,
                adapter.score_windows,
                "q",
                rows,
                phase=phase,
                on_submit=attempts.append,
            )
            try:
                assert entered.wait(0.5), "independent sources did not overlap"
                assert len(calls) == width
            finally:
                release.set()
            result = future.result(2)
        assert maximum[0] == width
        assert result.status == "complete"
        assert result.candidate_order == tuple(row.pk for row in rows)
        assert [s.chunk_pk for s in result.scores] == list(range(1, 9))
        assert [s.value for s in result.scores] == list(range(1, 9))
        assert all(s.window_coverage.is_complete() for s in result.scores)
        assert len(calls) == budget.pairs_used[phase] == sum(attempts) == 8
        assert budget._inflight == 0
        before = budget.pairs_used.copy()
        assert adapter.score_windows("q", rows, phase=phase) == result
        assert budget.pairs_used == before and len(calls) == 8
    finally:
        release.set()
        context.reset(token)
        budget.close("test_complete")


def test_out_of_order_sources_keep_original_window_and_score_identity(monkeypatch):
    monkeypatch.setenv("APP_RERANK_SCORE_CONCURRENCY", "3")
    budget = TurnBudget(TurnLimits())
    entered, release = Event(), Event()
    completion, attempts = [], []

    class Provider:
        def score_pair(self, pair, timeout):
            if pair[1].startswith("first"):
                assert release.wait(1)
            else:
                entered.set()
                release.set()
            completion.append(pair[1])
            return float(len(pair[1])), pair

    rows = chunks(3)
    rows[0].content = "first source"
    rows[1].content = "second " + "x" * 1600
    rows[2].content = "third source"
    adapter = WindowSelectionScorer(
        Provider(), budget=budget, pair_counter=lambda q, d: len(q) + len(d) + 8
    )
    try:
        result = adapter.score_windows(
            "q", rows, phase="acquisition", on_submit=attempts.append
        )
        assert entered.is_set() and not completion[0].startswith("first")
        assert result.status == "complete"
        assert result.candidate_order == (1, 2, 3)
        for row, score in zip(rows, result.scores, strict=True):
            plan = adapter.plan("q", row)
            assert score.chunk_pk == row.pk
            assert score.effective_pair_fingerprint == plan.fingerprint
            assert score.window_coverage.required_window_ids == plan.required_window_ids
            assert score.value == max(len(window.text) for window in plan.windows)
        assert sum(attempts) == budget.pairs_used["acquisition"] == len(completion)
    finally:
        release.set()
        budget.close("test_complete")
