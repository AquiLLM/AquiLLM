"""Bind private rerank scores to the final authorized search rows."""

from __future__ import annotations

from dataclasses import replace

from apps.documents.services.chunk_rerank_results import (
    RerankScoreSet,
    fingerprint_text,
)


def score_set_for_authorized_rows(
    score_set: RerankScoreSet, rows: tuple[object, ...]
) -> RerankScoreSet:
    """Discard score identities not present in the current authorized handoff."""
    current = {
        row.pk: (row.doc_id, row.chunk_number, fingerprint_text(row.content))
        for row in rows
    }
    if score_set.status == "complete" and score_set.scoring_kind != "rank_only":
        scores = tuple(
            score
            for score in score_set.scores
            if current.get(score.chunk_pk)
            == (score.document_id, score.chunk_number, score.source_fingerprint)
        )
        eligible_ids = {score.chunk_pk for score in scores}
        return replace(
            score_set,
            candidate_order=tuple(
                pk for pk in score_set.candidate_order if pk in eligible_ids
            ),
            scores=scores,
        )
    return replace(
        score_set,
        candidate_order=tuple(pk for pk in score_set.candidate_order if pk in current),
        scores=(),
    )


__all__ = ["score_set_for_authorized_rows"]
