"""Fail-closed reauthorization at chunk-search row handoff."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

from apps.documents.services.hybrid_graph_authorization import (
    is_exact_authorization_context,
    reauthorized_baseline,
)


def authorized_search_rows(
    rows: object, *, authorization_context: object | None, hybrid_requested: bool
) -> tuple[object, ...]:
    from .chunk_search_scope import current_scope
    from .chunk_search_scoped_rows import _closed_authorized_rows, validate_rows

    scope = current_scope()

    def reauthorize(current):
        if hybrid_requested or is_exact_authorization_context(authorization_context):
            try:
                return reauthorized_baseline(tuple(current), authorization_context)[0]
            except Exception:
                return ()
        return current

    if scope:
        return _closed_authorized_rows(rows, scope=scope, reauthorize=reauthorize)
    return tuple(reauthorize(validate_rows(rows)))


def finalize_search_rows(snapshot, ranked, *, authorize, pool=(), fallback_limit=None):
    """Finalize every returned channel before score/diagnostic serialization."""
    from .chunk_search_scope import current_scope

    groups = (
        tuple(snapshot.vector_results),
        tuple(snapshot.trigram_results),
        tuple(ranked),
        tuple(snapshot.full_text_results),
        tuple(snapshot.exact_results),
        tuple(pool),
    )
    retained = {
        id(row) for row in authorize(tuple(row for group in groups for row in group))
    }
    scope = current_scope()
    with scope.database() if scope else nullcontext():
        keys = set(scope.validate()) if scope else set()
        from .lexical_readiness import valid_full_text_ids

        full_text_ids = (
            valid_full_text_ids(scope, tuple(row for group in groups for row in group))
            if scope
            else set()
        )
        from .chunk_search_revalidation import withdraw_full_text
        from .lexical_readiness import current_full_text_coverage

        coverage = snapshot.full_text_coverage
        if scope:
            withdraw_full_text(
                scope, tuple(row for group in groups for row in group), full_text_ids
            )
            if coverage is not None and coverage.status != "disabled":
                coverage = current_full_text_coverage(scope)
            scope.validate()

    def keep(row, *, dense_only=False):
        if id(row) not in retained:
            return False
        dense = scope.dense_rows.get(row.pk) if scope else None
        if (
            scope
            and row.pk in scope.fts_rows
            and row.pk not in full_text_ids
            and row.pk not in scope.lexical_ids
            and (dense is None or dense.embedding_generation_key not in keys)
        ):
            return False
        return (
            not dense
            or dense.embedding_generation_key in keys
            or (not dense_only and row.pk in scope.lexical_ids | full_text_ids)
        )

    def retain(rows, *, dense_only=False):
        kept = tuple(row for row in rows if keep(row, dense_only=dense_only))
        return rows if len(kept) == len(rows) else kept

    vectors = retain(groups[0], dense_only=True)
    trigrams = retain(groups[1])
    full_text = tuple(
        row for row in retain(groups[3]) if scope is None or row.pk in full_text_ids
    )
    exact = retain(groups[4])
    pool = retain(groups[5])
    from .chunk_search_revalidation import final_fusion, reconcile_candidate_order

    fusion = final_fusion(
        scope,
        snapshot,
        pool,
        full_text_ids={r.pk for r in full_text},
        dense_ids={r.pk for r in vectors},
    )
    final_ranked = [row for row in groups[2] if keep(row)]
    if scope and scope.fts_invalid_ids and fallback_limit is not None:
        final_ranked = list(reconcile_candidate_order(pool)[:fallback_limit])
    return replace(
        snapshot,
        vector_results=vectors,
        vector_chunk_ids=tuple(row.pk for row in vectors),
        trigram_results=trigrams,
        trigram_chunk_ids=tuple(row.pk for row in trigrams),
        full_text_results=full_text,
        full_text_chunk_ids=tuple(row.pk for row in full_text),
        full_text_coverage=coverage,
        exact_results=exact,
        exact_chunk_ids=tuple(row.pk for row in exact),
        baseline_candidates=reconcile_candidate_order(
            retain(snapshot.baseline_candidates)
        ),
        fusion_v2=fusion,
    ), (vectors, trigrams, final_ranked)


def finalize_ranking(snapshot, ranking, *, authorize, query, top_k):
    """Keep final union authorization and score serialization in one handoff."""
    from .chunk_rerank_results import rank_only_result_for_chunks
    from .chunk_search_score_handoff import score_set_for_authorized_rows

    ranked = tuple(ranking.ranked_results)
    status = (
        ranking.score_set.status
        if ranking.score_set is not None
        else ranking.rerank_status
    )
    snapshot, final_rows = finalize_search_rows(
        snapshot,
        ranked,
        authorize=authorize,
        pool=ranking.combined_candidates,
        fallback_limit=top_k if status != "complete" else None,
    )
    rows = tuple(final_rows[2])
    score_set = ranking.score_set
    if score_set is not None:
        score_set = (
            score_set_for_authorized_rows(score_set, rows)
            if status == "complete"
            else rank_only_result_for_chunks(query, rows, top_k).score_set
        )
    return snapshot, final_rows, score_set


__all__ = ["authorized_search_rows"]
