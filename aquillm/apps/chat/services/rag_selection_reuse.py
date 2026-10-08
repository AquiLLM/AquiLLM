"""Exact-input acquisition score reuse, shared by both text modes."""

from collections.abc import Mapping
from math import isfinite

from apps.chat.services.rag_retrieval import FusedRetrievalPool
from apps.chat.services.rag_selection_inputs import FinalEvidenceInput
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


def retained_scores_match(
    retained,
    scores: Mapping[int, PassageScore],
    final_inputs: Mapping[int, FinalEvidenceInput],
) -> bool:
    """Check that every retained passage still has its exact scored input."""
    return all(
        item.chunk.pk in scores
        and scores[item.chunk.pk].document_id == item.chunk.doc_id
        and scores[item.chunk.pk].chunk_number == item.chunk.chunk_number
        and scores[item.chunk.pk].source_fingerprint
        == (
            final_inputs[item.chunk.pk].source_fingerprint
            if final_inputs
            else fingerprint_text(item.chunk.content)
        )
        and (
            scores[item.chunk.pk].effective_pair_fingerprint
            == final_inputs[item.chunk.pk].pair_fingerprint
            if final_inputs
            else True
        )
        and isfinite(scores[item.chunk.pk].value)
        for item in retained
    )


def _reused_scores(
    *,
    pool: FusedRetrievalPool,
    hydrated,
    query: str,
    scorer: SelectionScorer,
    final_inputs: Mapping[int, FinalEvidenceInput] | None = None,
) -> dict[int, PassageScore]:
    by_id = {item.chunk.pk: item for item in hydrated}

    def pair_fingerprint(item):
        return (
            final_inputs[item.chunk.pk].pair_fingerprint
            if final_inputs is not None
            else expected_score_fingerprint(scorer, query, item.chunk)
        )

    def identity(item):
        if final_inputs is not None:
            prepared = final_inputs[item.chunk.pk]
            return (
                item.chunk.pk,
                item.chunk.doc_id,
                item.chunk.chunk_number,
                prepared.source_fingerprint,
                prepared.pair_fingerprint,
            )
        return score_identity_for_chunk(
            item.chunk,
            effective_pair_fingerprint=pair_fingerprint(item),
        )

    expected_query = fingerprint_text(query)
    reused: dict[int, PassageScore] = {}
    if scorer.scoring_kind == "listwise":
        complete_order = tuple(item.chunk.pk for item in hydrated)
        canonical = tuple(
            (
                item.chunk.pk,
                final_inputs[item.chunk.pk].source_fingerprint
                if final_inputs is not None
                else item.source_fingerprint,
                pair_fingerprint(item),
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
        identities = tuple(identity(by_id[pk]) for pk in score_set.candidate_order)
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
