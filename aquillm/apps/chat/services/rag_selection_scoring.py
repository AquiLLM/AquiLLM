"""Build one authorized pool with comparable model scores or whole-pool RRF."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from time import monotonic

from apps.chat.services.rag_retrieval import FusedRetrievalPool
from apps.chat.services.rag_selection_hydration import hydrate_pool_rows
from apps.chat.services.rag_selection_types import SelectionCandidate
from apps.collections.services.retrieval_authorization import (
    RetrievalAuthorizationContext,
)
from apps.documents.services.chunk_rerank_config import rerank_score_concurrency
from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    fingerprint_pair,
    fingerprint_pool,
    fingerprint_text,
    score_identity_for_chunk,
    validate_score_set,
)
from apps.documents.services.chunk_rerank_scoring import (
    SelectionScorer,
    score_missing_pairs,
)


@dataclass(frozen=True)
class PreparedSelection:
    candidates: tuple[SelectionCandidate, ...]
    score_status: str
    reused_pairs: int
    new_pairs: int
    scoring_duration_ms: float
    fallback_reason: str | None


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return ()
    low, high = min(values), max(values)
    if high - low <= 1e-9 * max(1.0, abs(low), abs(high)):
        return (0.5,) * len(values)
    return tuple((value - low) / (high - low) for value in values)


def _reused_scores(
    *, pool: FusedRetrievalPool, hydrated, query: str, scorer: SelectionScorer
) -> dict[int, PassageScore]:
    by_id = {item.chunk.pk: item for item in hydrated}
    expected_query = fingerprint_text(query)
    reused: dict[int, PassageScore] = {}
    if scorer.scoring_kind == "listwise":
        complete_order = tuple(item.chunk.pk for item in hydrated)
        canonical = tuple(
            (
                item.chunk.pk,
                item.source_fingerprint,
                fingerprint_pair(*scorer.prepare_pair(query, item.chunk)),
            )
            for item in hydrated
        )
        expected_pool = fingerprint_pool(canonical)
    for score_set in pool.source_score_sets:
        if (
            score_set.status != "complete"
            or score_set.scoring_kind != scorer.scoring_kind
            or score_set.query_fingerprint != expected_query
            or score_set.scorer_fingerprint != scorer.scorer_fingerprint
        ):
            continue
        if scorer.scoring_kind == "listwise" and (
            score_set.candidate_order != complete_order
            or score_set.pool_fingerprint != expected_pool
        ):
            continue
        if any(pk not in by_id for pk in score_set.candidate_order):
            continue
        identities = tuple(
            score_identity_for_chunk(
                by_id[pk].chunk,
                effective_pair_fingerprint=fingerprint_pair(
                    *scorer.prepare_pair(query, by_id[pk].chunk)
                ),
            )
            for pk in score_set.candidate_order
        )
        try:
            validate_score_set(
                score_set,
                authorized_identities=identities,
                expected_query_fingerprint=expected_query,
                expected_scorer_fingerprint=scorer.scorer_fingerprint,
            )
        except (TypeError, ValueError):
            continue
        reused.update((score.chunk_pk, score) for score in score_set.scores)
        if scorer.scoring_kind == "listwise":
            break
    return reused


def prepare_selection_candidates(
    *,
    pool: FusedRetrievalPool,
    primary_query: str,
    authorization: RetrievalAuthorizationContext,
    deadline: float,
    allow_new_scores: bool,
    scorer: SelectionScorer | None = None,
    chunk_loader=None,
    clock: Callable[[], float] = monotonic,
) -> PreparedSelection:
    """Never widen pool membership or mix model and rank-derived scales."""
    if len(pool.rows) > 45:
        raise ValueError("candidate union exceeds the hard cap")
    first = hydrate_pool_rows(pool.rows, authorization, chunk_loader=chunk_loader)
    initial_by_id = {item.chunk.pk: item.source_fingerprint for item in first}
    used_scorer = scorer
    if used_scorer is None and allow_new_scores:
        from apps.documents.services.chunk_rerank_selection_provider import (
            current_selection_scorer,
        )

        used_scorer = current_selection_scorer(deadline=deadline, clock=clock)
    reused: dict[int, PassageScore] = {}
    new_scores: dict[int, PassageScore] = {}
    new_pairs = 0
    score_start = clock()
    fallback_reason = None
    if used_scorer is None:
        fallback_reason = "scorer_unavailable"
    elif first:
        try:
            reused = _reused_scores(
                pool=pool,
                hydrated=first,
                query=primary_query,
                scorer=used_scorer,
            )
            missing = tuple(item.chunk for item in first if item.chunk.pk not in reused)
            if missing and allow_new_scores and clock() < deadline:
                # Listwise scores depend on the complete ordered pool.
                to_score = (
                    tuple(item.chunk for item in first)
                    if (used_scorer.scoring_kind == "listwise")
                    else missing
                )
                submitted: list[int] = []
                fresh = score_missing_pairs(
                    query=primary_query,
                    chunks=to_score,
                    scorer=used_scorer,
                    deadline=deadline,
                    max_inflight=min(6, rerank_score_concurrency()),
                    clock=clock,
                    on_submit=lambda count: submitted.append(count),
                )
                new_pairs = sum(submitted)
                if fresh.status == "complete" and (
                    fresh.scorer_fingerprint == used_scorer.scorer_fingerprint
                    and fresh.query_fingerprint == fingerprint_text(primary_query)
                    and fresh.scoring_kind == used_scorer.scoring_kind
                ):
                    if fresh.candidate_order != tuple(chunk.pk for chunk in to_score):
                        raise ValueError("new score pool changed")
                    validate_score_set(
                        fresh,
                        authorized_identities=tuple(
                            score_identity_for_chunk(
                                chunk,
                                effective_pair_fingerprint=fingerprint_pair(
                                    *used_scorer.prepare_pair(primary_query, chunk)
                                ),
                            )
                            for chunk in to_score
                        ),
                        expected_query_fingerprint=fingerprint_text(primary_query),
                        expected_scorer_fingerprint=used_scorer.scorer_fingerprint,
                    )
                    new_scores = {score.chunk_pk: score for score in fresh.scores}
            if missing and not new_scores and not allow_new_scores:
                fallback_reason = "new_scores_disabled"
        except (TypeError, ValueError):
            fallback_reason = "score_incompatible"
    scoring_duration_ms = max(0.0, (clock() - score_start) * 1000.0)

    # Permissions and content may change while inference is running.
    current = hydrate_pool_rows(pool.rows, authorization, chunk_loader=chunk_loader)
    retained = tuple(
        item
        for item in current
        if initial_by_id.get(item.chunk.pk) == item.source_fingerprint
    )
    scores = {**reused, **new_scores}
    model_complete = (
        bool(retained)
        and used_scorer is not None
        and all(
            item.chunk.pk in scores
            and scores[item.chunk.pk].document_id == item.chunk.doc_id
            and scores[item.chunk.pk].chunk_number == item.chunk.chunk_number
            and scores[item.chunk.pk].source_fingerprint == item.source_fingerprint
            and isfinite(scores[item.chunk.pk].value)
            for item in retained
        )
        and clock() < deadline
    )
    if model_complete:
        relevance = _normalize(tuple(scores[item.chunk.pk].value for item in retained))
        status = "model"
    else:
        by_citation = dict(pool.fused_scores)
        relevance = _normalize(
            tuple(
                by_citation.get(str(item.row.get("citation", item.row.get("ref"))), 0.0)
                for item in retained
            )
        )
        status = "rank_fallback"
        fallback_reason = fallback_reason or (
            "deadline" if clock() >= deadline else "incomplete_scores"
        )
    rank_by_id = {
        row.get("chunk_id", row.get("i")): rank
        for rank, row in enumerate(pool.rows, start=1)
    }
    candidates = tuple(
        SelectionCandidate(
            item.chunk.pk,
            str(item.chunk.doc_id),
            item.chunk.chunk_number,
            item.excerpt,
            value,
            rank_by_id[item.chunk.pk],
            item.source_fingerprint,
            item.row,
        )
        for item, value in zip(retained, relevance)
    )
    return PreparedSelection(
        candidates,
        status,
        len(reused),
        new_pairs,
        scoring_duration_ms,
        fallback_reason,
    )


__all__ = ["PreparedSelection", "prepare_selection_candidates"]
