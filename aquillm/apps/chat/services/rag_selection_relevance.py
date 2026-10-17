"""Comparable whole-pool relevance without treating missing scores as low scores."""

from .rag_selection_policy import normalize_relevance_scores as normalize
from .rag_selection_reuse import retained_scores_match


def _partial_rank_relevance(retained, scores):
    """Reorder scored sources within their original slots; unknown slots stay put.

    Only model scores from the same query/provider are compared. The selector
    then receives one ordinal scale for the entire pool, not a mixture of model
    magnitudes and fused retrieval scores. No retrieval lane owns a reserved slot.
    """
    ranked = iter(sorted(
        (item for item in retained if item.chunk.pk in scores),
        key=lambda item: -scores[item.chunk.pk].value,
    ))
    order = tuple(
        next(ranked).chunk.pk if item.chunk.pk in scores else item.chunk.pk
        for item in retained
    )
    positions = {pk: len(order) - index for index, pk in enumerate(order)}
    return normalize(tuple(positions[item.chunk.pk] for item in retained))


def selection_relevance(
    *, pool, retained, admitted, scores, final_inputs, scorer,
    live, allow_partial, fallback_reason,
):
    """Return relevance and only the scores whose provenance may be published."""
    if live and retained and scorer is not None:
        if (
            scorer.scoring_kind in {"pointwise", "listwise"}
            and retained_scores_match(retained, scores, final_inputs)
        ):
            return (
                normalize(tuple(scores[item.chunk.pk].value for item in retained)),
                "model", fallback_reason, scores,
            )
        scored = tuple(item for item in retained if item.chunk.pk in scores)
        if (
            allow_partial and fallback_reason is None and scored
            and scorer.scoring_kind == "pointwise"
            and all(item.chunk.pk in scores for item in admitted)
            and retained_scores_match(scored, scores, final_inputs)
        ):
            return (
                _partial_rank_relevance(retained, scores),
                "partial_model_ranked", "incomplete_scores", scores,
            )
    by_citation = dict(pool.fused_scores)
    return (
        normalize(tuple(
            by_citation.get(str(item.row.get("citation", item.row.get("ref"))), 0.0)
            for item in retained
        )),
        "rank_fallback", fallback_reason, {},
    )
