"""Rehydrate public retrieval rows from the current authorized source."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from uuid import UUID

from apps.chat.consumers.utils import truncate_tool_text
from apps.chat.services.rag_selection_types import SelectionCandidate
from apps.collections.services.retrieval_authorization import (
    RetrievalAuthorizationContext,
    revalidate_retrieval_authorization_context,
)
from apps.documents.models.chunks import TextChunk
from apps.documents.services.chunk_rerank_results import fingerprint_text

type ChunkLoader = Callable[
    [RetrievalAuthorizationContext, tuple[int, ...]], Iterable[TextChunk]
]


@dataclass(frozen=True)
class HydratedRow:
    row: Mapping[str, object]
    chunk: TextChunk
    excerpt: str
    source_fingerprint: str


def _current_chunks(
    authorization: RetrievalAuthorizationContext, ids: tuple[int, ...]
) -> Iterable[TextChunk]:
    return TextChunk.objects.using(authorization.database_alias).filter(pk__in=ids)


def hydrate_pool_rows(
    rows: tuple[Mapping[str, object], ...],
    authorization: RetrievalAuthorizationContext,
    *,
    chunk_loader: ChunkLoader | None = None,
) -> tuple[HydratedRow, ...]:
    """Keep only rows matching current permission, coordinates and emitted excerpt."""
    scope = revalidate_retrieval_authorization_context(context=authorization)
    if not scope.document_ids:
        return ()
    ids = tuple(
        value
        for row in rows
        if type(value := row.get("chunk_id", row.get("i"))) is int and value > 0
    )
    if not ids:
        return ()
    loader = chunk_loader or _current_chunks
    current = {chunk.pk: chunk for chunk in loader(authorization, ids)}
    allowed = frozenset(scope.document_ids)
    hydrated = []
    for row in rows:
        pk = row.get("chunk_id", row.get("i"))
        doc = row.get("doc_id", row.get("d"))
        number = row.get("chunk", row.get("c"))
        excerpt = row.get("text", row.get("x"))
        citation = row.get("citation", row.get("ref"))
        if (
            type(pk) is not int
            or type(doc) is not str
            or type(number) is not int
            or type(excerpt) is not str
            or type(citation) is not str
        ):
            continue
        chunk = current.get(pk)
        if chunk is None or not isinstance(chunk.doc_id, UUID):
            continue
        if (
            chunk.doc_id not in allowed
            or str(chunk.doc_id) != doc
            or chunk.chunk_number != number
            or citation != f"[doc:{doc} chunk:{pk}]"
            or excerpt != truncate_tool_text(chunk.content)
        ):
            continue
        hydrated.append(
            HydratedRow(row, chunk, excerpt, fingerprint_text(chunk.content))
        )
    return tuple(hydrated)


def revalidate_selection_candidates(
    candidates: tuple[SelectionCandidate, ...],
    authorization: RetrievalAuthorizationContext,
    *,
    chunk_loader: ChunkLoader | None = None,
) -> tuple[SelectionCandidate, ...]:
    """Preserve selected order while dropping revoked or changed source rows."""
    hydrated = hydrate_pool_rows(
        tuple(candidate.row for candidate in candidates),
        authorization,
        chunk_loader=chunk_loader,
    )
    by_id = {item.chunk.pk: item for item in hydrated}
    return tuple(
        candidate
        for candidate in candidates
        if (item := by_id.get(candidate.chunk_id)) is not None
        and item.source_fingerprint == candidate.source_fingerprint
        and item.excerpt == candidate.text
        and str(item.chunk.doc_id) == candidate.doc_id
        and item.chunk.chunk_number == candidate.chunk_number
    )


__all__ = ["HydratedRow", "hydrate_pool_rows", "revalidate_selection_candidates"]
