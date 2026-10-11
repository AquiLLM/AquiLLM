"""Narrow source-role adapters for persisted projection representation authority."""
from uuid import UUID
from types import SimpleNamespace

from django.db import connections

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from .limits import MAX_DETAIL_ROWS
from .representation_binding import validate_record_binding


def parameters(projection, *, purpose, lease_owner):
    if type(projection.id) is not UUID or type(projection.generation_key) is not UUID:
        raise TypeError("projection identity must contain exact UUIDs")
    validate_record_binding(projection)
    if purpose not in {"build", "audit"}:
        raise ValueError("projection read purpose is invalid")
    if purpose == "build" and (
        type(lease_owner) is not str or not 1 <= len(lease_owner) <= 128
        or lease_owner != lease_owner.strip()
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
    if purpose == "audit" and lease_owner is not None:
        raise SourceScopeUnavailable("source_scope_unavailable")
    return (projection.id, projection.generation_key,
            projection.representation_digest, purpose, lease_owner)


def validate_current(projection, *, using, purpose, lease_owner):
    values = parameters(projection, purpose=purpose, lease_owner=lease_owner)
    with connections[using].cursor() as cursor:
        cursor.execute("SELECT public.kg_projection_representation_current_v1(%s,%s,%s,%s,%s)",
                       values)
        row = cursor.fetchone()
    if row != (True,):
        raise SourceScopeUnavailable("source_scope_unavailable")


def current_coordinates(projection, *, using, purpose, lease_owner):
    values = parameters(projection, purpose=purpose, lease_owner=lease_owner)
    with connections[using].cursor() as cursor:
        cursor.execute("SELECT id,doc_id,chunk_number FROM "
                       "public.kg_projection_representation_chunks_v1(%s,%s,%s,%s,%s)", values)
        rows = cursor.fetchmany(MAX_DETAIL_ROWS + 1)
    if len(rows) > MAX_DETAIL_ROWS:
        raise SourceScopeUnavailable("projection_chunk_limit")
    return tuple(dict(zip(("id", "doc_id", "chunk_number"), row, strict=True)) for row in rows)


def projection_chunk_rows(projection, document_ids, *, using, purpose, lease_owner, batch_size):
    if purpose == "prune":
        from apps.documents.models import TextChunk
        from .limits import bounded_projection_rows

        return bounded_projection_rows(TextChunk.objects.using(using).filter(doc_id__in=document_ids),
                                       ("id", "doc_id", "chunk_number"), batch_size)
    return current_coordinates(
        SimpleNamespace(**projection, collection_pk_snapshot=projection["collection_id"]),
        using=using, purpose=purpose, lease_owner=lease_owner)
