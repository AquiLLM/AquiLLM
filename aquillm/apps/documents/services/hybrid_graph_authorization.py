"""Authorization and adapter seams for projected hybrid graph retrieval."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from uuid import UUID

_DATABASE_ID_MAX = 9_223_372_036_854_775_807


@dataclass(frozen=True, slots=True)
class HybridGraphRetrievalDependencies:
    """Request-independent adapters required by the projected graph overlay."""

    runtime: object = field(repr=False)
    settings: object = field(repr=False)
    materialize: Callable[..., tuple[object, ...]] = field(repr=False)

    def __post_init__(self) -> None:
        operations = (
            "prepare_shared",
            "run_direct",
            "prepare_extended",
            "run_extended",
        )
        if any(not callable(getattr(self.runtime, name, None)) for name in operations):
            raise TypeError("runtime must implement every hybrid branch operation")
        if self.settings is None:
            raise TypeError("settings must be non-null")
        if not callable(self.materialize):
            raise TypeError("materialize must be callable")


def is_exact_authorization_context(value: object) -> bool:
    from apps.collections.services.retrieval_authorization import (
        RetrievalAuthorizationContext,
    )

    return type(value) is RetrievalAuthorizationContext


def documents_match_retrieval_authorization(
    documents: Iterable[object], authorization: object
) -> bool:
    """Compare exact document scalars with the frozen selected scope."""

    if not is_exact_authorization_context(authorization):
        return False
    document_ids: set[UUID] = set()
    collection_ids: set[int] = set()
    try:
        for document in documents:
            document_id = getattr(document, "id", None)
            collection_id = getattr(document, "collection_id", None)
            if (
                type(document_id) is not UUID
                or type(collection_id) is not int
                or not 1 <= collection_id <= _DATABASE_ID_MAX
                or document_id in document_ids
            ):
                return False
            document_ids.add(document_id)
            collection_ids.add(collection_id)
            if len(document_ids) > 10_000 or len(collection_ids) > 128:
                return False
    except TypeError:
        return False
    return (
        frozenset(document_ids) == authorization.selected_document_ids
        and frozenset(collection_ids) == authorization.selected_collection_ids
    )


def reauthorized_baseline(
    baseline: tuple[object, ...], authorization: object
) -> tuple[tuple[object, ...], bool]:
    """Retain current frozen rows and report whether graph ranks remain usable."""

    from apps.collections.services.retrieval_authorization import (
        RetrievalAuthorizationContext,
        revalidate_retrieval_authorization_context,
    )

    if type(authorization) is not RetrievalAuthorizationContext:
        raise TypeError("authorization must be an exact retrieval context")
    current = revalidate_retrieval_authorization_context(context=authorization)
    allowed_documents = frozenset(current.document_ids)
    retained = tuple(
        row for row in baseline if getattr(row, "doc_id", None) in allowed_documents
    )
    graph_allowed = (
        frozenset(current.collection_ids) == authorization.selected_collection_ids
        and allowed_documents == authorization.selected_document_ids
    )
    return retained, graph_allowed


__all__ = [
    "HybridGraphRetrievalDependencies",
    "documents_match_retrieval_authorization",
    "is_exact_authorization_context",
    "reauthorized_baseline",
]
