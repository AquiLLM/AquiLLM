"""Exact-input acquisition score reuse, shared by both text modes."""

from apps.chat.services.rag_retrieval import FusedRetrievalPool
from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    fingerprint_pool,
    fingerprint_text,
    score_identity_for_chunk,
    validate_score_set,
)
from apps.documents.services.chunk_rerank_scoring import SelectionScorer
from apps.documents.services.chunk_rerank_window_adapter import (
    compatible_window_score,
    expected_score_fingerprint,
)


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
                expected_score_fingerprint(scorer, query, item.chunk),
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
                effective_pair_fingerprint=expected_score_fingerprint(
                    scorer, query, by_id[pk].chunk
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
        if any(
            not compatible_window_score(
                scorer, score, query, by_id[score.chunk_pk].chunk
            )
            for score in score_set.scores
        ):
            continue
        reused.update((score.chunk_pk, score) for score in score_set.scores)
        if scorer.scoring_kind == "listwise":
            break
    return reused
