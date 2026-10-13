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
from lib.retrieval.evidence import PreparedEvidence

type ChunkLoader = Callable[
    [RetrievalAuthorizationContext, tuple[int, ...]], Iterable[TextChunk]
]


@dataclass(frozen=True)
class HydratedRow:
    row: Mapping[str, object]
    chunk: TextChunk
    excerpt: str
    source_fingerprint: str
    prepared_evidence: PreparedEvidence | None = None


def _current_chunks(
    authorization: RetrievalAuthorizationContext, ids: tuple[int, ...]
) -> Iterable[TextChunk]:
    from apps.documents.services.source_loading import (
        current_source_runtime,
        source_query_rows,
    )

    query = TextChunk.objects.using(authorization.database_alias).filter(pk__in=ids)
    if current_source_runtime():
        return source_query_rows(query)
    from .rag_publication_database import _bounded_database
    from .rag_representation_publication import authorized_snapshot
    from apps.documents.services.representation_store import active_chunk_scope, validate_representation_snapshot
    from lib.llm.turn_context import current_turn
    turn = current_turn()
    from apps.documents.services.representation_manual_scope import private_acquisition_runtime
    runtime = private_acquisition_runtime()
    if runtime is not None:
        from apps.documents.services.chunk_search_scope import validate_requested_authorization
        from apps.documents.services.representation_tool_scope import tool_scope
        from apps.documents.services.representation_contracts import SourceScopeUnavailable
        from apps.documents.services.source_loading import bounded_source_database
        if (runtime.budget is not turn.budget or runtime.authorization is not
                getattr(runtime, "_media_authorization", None)):
            raise SourceScopeUnavailable("source_scope_unavailable")
        with runtime.lock, bounded_source_database(runtime, query.db):
            scope = tool_scope(runtime, using=query.db, require_existing=True)
            documents = tuple(item.document for item in scope.representation.selections
                              if item.document.id in authorization.selected_document_ids)
            validate_requested_authorization(runtime, authorization, documents)
            return source_query_rows(query, runtime=runtime, require_existing=True)
    with _bounded_database(query.db, min(3, turn.budget.remaining_ms() / 1000) if turn else 3):
        snapshot = authorized_snapshot(authorization, using=query.db)
        rows = tuple(active_chunk_scope(query, snapshot=snapshot))
        validate_representation_snapshot(snapshot, using=query.db)
        return rows


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
    if any(c.prepared_evidence is not None for c in candidates):
        from apps.chat.services.rag_source_hydration import revalidate_source_candidates

        return revalidate_source_candidates(
            candidates, authorization, chunk_loader=chunk_loader
        )
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
