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
    from .chunk_search_scoped_rows import validate_rows

    scope = current_scope()
    with scope.database() if scope else nullcontext():
        rows = validate_rows(rows)
        if hybrid_requested or is_exact_authorization_context(authorization_context):
            try:
                rows = reauthorized_baseline(tuple(rows), authorization_context)[0]
            except Exception:
                rows = ()
        if scope:
            rows = validate_rows(rows)
            scope.validate()  # Empty results must also obey the original deadline.
        return tuple(rows)


def finalize_search_rows(snapshot, ranked, *, authorize):
    """Finalize every returned channel before score/diagnostic serialization."""
    from .chunk_search_scope import current_scope

    groups = (
        tuple(snapshot.vector_results),
        tuple(snapshot.trigram_results),
        tuple(ranked),
        tuple(snapshot.full_text_results),
        tuple(snapshot.exact_results),
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
    return replace(
        snapshot,
        vector_results=vectors,
        vector_chunk_ids=tuple(row.pk for row in vectors),
        trigram_results=trigrams,
        trigram_chunk_ids=tuple(row.pk for row in trigrams),
        full_text_results=retain(groups[3]),
        full_text_chunk_ids=tuple(row.pk for row in retain(groups[3])),
        exact_results=retain(groups[4]),
        exact_chunk_ids=tuple(row.pk for row in retain(groups[4])),
    ), (vectors, trigrams, [row for row in groups[2] if keep(row)])


__all__ = ["authorized_search_rows"]
