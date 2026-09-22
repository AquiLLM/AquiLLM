"""Deadline-bound additional scoring for one current, authorized candidate pool."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from math import isfinite
from time import monotonic
from typing import Protocol

from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    fingerprint_pair,
    fingerprint_pool,
    fingerprint_text,
)

type Pair = tuple[str, str]
type ScoredPair = tuple[float, Pair]


class SelectionScorer(Protocol):
    scoring_kind: str
    scorer_fingerprint: str

    def prepare_pair(self, query: str, chunk: object) -> Pair: ...

    def score_pair(self, pair: Pair, timeout_seconds: float) -> ScoredPair | None: ...

    def score_pool(
        self, pairs: tuple[Pair, ...], timeout_seconds: float
    ) -> tuple[ScoredPair, ...] | None: ...


def score_missing_pairs(
    *,
    query: str,
    chunks,
    scorer: SelectionScorer,
    deadline: float,
    max_pairs: int = 45,
    max_inflight: int = 6,
    clock: Callable[[], float] = monotonic,
    on_submit: Callable[[int], None] | None = None,
) -> RerankScoreSet:
    """Score at most one bounded pool; late futures cannot publish a result."""
    rows = tuple(chunks)
    if (
        len(rows) > 45
        or len(rows) > max_pairs
        or len({row.pk for row in rows}) != len(rows)
    ):
        rows = ()
        oversized = True
    else:
        oversized = False
    order = tuple(row.pk for row in rows)
    kind = getattr(scorer, "scoring_kind", "rank_only")
    scorer_fp = getattr(scorer, "scorer_fingerprint", "")
    try:
        pairs = tuple(scorer.prepare_pair(query, row) for row in rows)
    except Exception:
        pairs = ()
        oversized = True
    identities = tuple(
        (row.pk, fingerprint_text(row.content), fingerprint_pair(*pair))
        for row, pair in zip(rows, pairs)
    )
    pool_fp = fingerprint_pool(identities)

    def unavailable() -> RerankScoreSet:
        return RerankScoreSet(
            "v2",
            fingerprint_text(query),
            scorer_fp,
            pool_fp,
            "rank_only",
            "unavailable",
            order,
            (),
        )

    if (
        oversized
        or not rows
        or len(pairs) != len(rows)
        or kind not in ("pointwise", "listwise")
        or not scorer_fp
        or max_inflight < 1
        or clock() >= deadline
    ):
        return unavailable()

    executor = ThreadPoolExecutor(max_workers=min(6, max_inflight, len(rows)))
    completed: dict[int, ScoredPair] = {}
    try:
        if kind == "listwise":
            pending = {}
            remaining = deadline - clock()
            if remaining > 0:
                future = executor.submit(scorer.score_pool, pairs, remaining)
                pending[future] = -1
                if on_submit is not None:
                    on_submit(len(rows))
        else:
            pending = {}
            next_index = 0

            def submit_available() -> None:
                nonlocal next_index
                while (
                    next_index < len(rows)
                    and len(pending) < min(6, max_inflight)
                    and clock() < deadline
                ):
                    index = next_index
                    next_index += 1
                    remaining = deadline - clock()
                    if remaining <= 0:
                        break
                    future = executor.submit(scorer.score_pair, pairs[index], remaining)
                    pending[future] = index
                    if on_submit is not None:
                        on_submit(1)

            submit_available()
        while pending and clock() < deadline:
            done, _ = wait(
                pending,
                timeout=max(0.0, deadline - clock()),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                break
            for future in done:
                index = pending.pop(future)
                try:
                    result = future.result()
                except Exception:
                    result = None
                if kind == "listwise":
                    if isinstance(result, tuple) and len(result) == len(rows):
                        completed.update(enumerate(result))
                elif result is not None:
                    completed[index] = result
            if kind == "pointwise":
                submit_available()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    if clock() >= deadline or len(completed) != len(rows):
        return unavailable()
    scores = []
    for index, row in enumerate(rows):
        result = completed[index]
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or type(result[0]) not in (int, float)
            or not isfinite(float(result[0]))
            or not isinstance(result[1], tuple)
            or len(result[1]) != 2
            or any(type(part) is not str for part in result[1])
        ):
            return unavailable()
        value, successful_pair = result
        scores.append(
            PassageScore(
                row.pk,
                row.doc_id,
                row.chunk_number,
                fingerprint_text(row.content),
                fingerprint_pair(*successful_pair),
                float(value),
            )
        )
    return RerankScoreSet(
        "v2",
        fingerprint_text(query),
        scorer_fp,
        pool_fp,
        kind,
        "complete",
        order,
        tuple(scores),
    )


__all__ = ["SelectionScorer", "score_missing_pairs"]
