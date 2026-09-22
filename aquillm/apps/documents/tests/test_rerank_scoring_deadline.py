"""Deadline and concurrency behavior of additional final-pool scoring."""

from __future__ import annotations

from threading import Event, Lock
from time import monotonic, sleep
from types import SimpleNamespace
from uuid import UUID


def test_timeout_returns_without_waiting_for_workers_or_publishing_late_scores():
    from apps.documents.services.chunk_rerank_scoring import score_missing_pairs

    gate = Event()
    lock = Lock()
    started = []

    class BlockingScorer:
        scoring_kind = "pointwise"
        scorer_fingerprint = "known-model"

        def prepare_pair(self, query, chunk):
            return query, chunk.content

        def score_pair(self, pair, timeout_seconds):
            with lock:
                started.append(pair)
            gate.wait(1.0)
            return 0.9, pair

    chunks = tuple(
        SimpleNamespace(
            pk=pk,
            doc_id=UUID("11111111-1111-4111-8111-111111111111"),
            chunk_number=pk - 1,
            content=f"content {pk}",
        )
        for pk in range(1, 20)
    )
    start = monotonic()
    submitted = []
    try:
        result = score_missing_pairs(
            query="question",
            chunks=chunks,
            scorer=BlockingScorer(),
            deadline=start + 0.05,
            max_inflight=6,
            on_submit=lambda count: submitted.append(count),
        )
        elapsed = monotonic() - start
        assert result.status == "unavailable" and result.scores == ()
        assert len(started) <= 6
        assert sum(submitted) <= 6
        assert sum(submitted) >= len(started)
        assert elapsed < 0.5
    finally:
        gate.set()
    sleep(0.02)
    assert result.status == "unavailable" and result.scores == ()


def test_expired_clock_never_submits_a_pair():
    from apps.documents.services.chunk_rerank_scoring import score_missing_pairs

    class ExpiredScorer:
        scoring_kind = "pointwise"
        scorer_fingerprint = "known-model"

        def prepare_pair(self, query, chunk):
            return query, chunk.content

        def score_pair(self, pair, timeout_seconds):
            raise AssertionError("worker submitted after deadline")

    chunk = SimpleNamespace(
        pk=1,
        doc_id=UUID("11111111-1111-4111-8111-111111111111"),
        chunk_number=0,
        content="source",
    )
    result = score_missing_pairs(
        query="question",
        chunks=(chunk,),
        scorer=ExpiredScorer(),
        deadline=5.0,
        clock=lambda: 5.0,
    )
    assert result.status == "unavailable"
