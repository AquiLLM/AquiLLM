"""Fail-closed reauthorization at chunk-search row handoff."""

from __future__ import annotations

from apps.documents.services.hybrid_graph_authorization import (
    is_exact_authorization_context,
    reauthorized_baseline,
)


def authorized_search_rows(
    rows: object, *, authorization_context: object | None, hybrid_requested: bool
) -> tuple[object, ...]:
    if not hybrid_requested and not is_exact_authorization_context(
        authorization_context
    ):
        return tuple(rows)
    try:
        return reauthorized_baseline(tuple(rows), authorization_context)[0]
    except Exception:
        return ()


__all__ = ["authorized_search_rows"]
