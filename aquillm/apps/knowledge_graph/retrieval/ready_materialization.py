"""Multi-projection private chunk materialization with current attestation."""

from __future__ import annotations

from uuid import UUID

from apps.knowledge_graph.projection.identifiers import (
    OpaqueProjectionKey,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.projection.records import PrivateProjectionChunkReferenceV1
from apps.knowledge_graph.retrieval.materialization import materialize_projected_chunks

from .ready_scope import SelectedReadyScopeV1

_MAX_SELECTED_CHUNKS = 40  # Two branches, each returning at most 20 candidates.
_MATERIALIZATION_BATCH_SIZE = 20


class DjangoPrivateChunkMapRepository:
    def __init__(self, source_using: str = "projection_source") -> None:
        self.source_using = source_using

    def locate(self, *, projection_ids: tuple[UUID, ...], chunk_keys: tuple[str, ...]):
        from apps.knowledge_graph.models import ProjectionChunkReference

        return tuple(
            ProjectionChunkReference.objects.using(self.source_using)
            .filter(
                projection_id__in=projection_ids,
                projection_chunk_key__in=chunk_keys,
            )
            .order_by("projection_id", "projection_chunk_key")
            .values_list("projection_id", "projection_chunk_key")
        )

    def load_private_chunk_map(
        self,
        *,
        projection_id,
        chunk_keys,
        expected_private_mapping_checksum,
        database_alias,
    ):
        del database_alias
        from apps.knowledge_graph.models import (
            CollectionGraphProjection,
            ProjectionChunkReference,
        )

        projection = CollectionGraphProjection.objects.using(self.source_using).get(
            pk=projection_id, state="ready"
        )
        if projection.private_mapping_checksum != expected_private_mapping_checksum:
            raise ValueError("private mapping authority checksum is stale")
        rows = tuple(
            ProjectionChunkReference.objects.using(self.source_using)
            .filter(projection_id=projection_id, projection_chunk_key__in=chunk_keys)
            .order_by("projection_chunk_key")
            .values(
                "projection_chunk_key",
                "integer_chunk_pk",
                "document_uuid",
                "chunk_number",
            )
        )
        return projection.private_mapping_checksum, tuple(
            PrivateProjectionChunkReferenceV1(
                row["projection_chunk_key"],
                row["integer_chunk_pk"],
                str(row["document_uuid"]),
                row["chunk_number"],
            )
            for row in rows
        )

    def load_chunk_objects(
        self, *, chunk_predicates, authorized_document_ids, database_alias
    ):
        from apps.documents.models import TextChunk
        from apps.documents.services.source_loading import source_query_rows

        pks = tuple(row[0] for row in chunk_predicates)
        return source_query_rows(
            TextChunk.objects.using(database_alias)
            .filter(pk__in=pks, doc_id__in=authorized_document_ids)
            .order_by("pk")
        )


def materialize_selected_ready_chunks(
    *, scope, chunk_keys, authorization, repository=None
):
    if type(scope) is not SelectedReadyScopeV1:
        raise TypeError("scope must be exact")
    if (
        type(chunk_keys) is not tuple
        or not chunk_keys
        or len(chunk_keys) > _MAX_SELECTED_CHUNKS
    ):
        raise ValueError(
            "chunk_keys must be a nonempty exact tuple within its union cap"
        )
    if any(
        type(key) is not OpaqueProjectionKey
        or key.domain is not ProjectionIdentifierDomain.CHUNK
        for key in chunk_keys
    ):
        raise TypeError("chunk_keys must contain exact opaque chunk keys")
    raw_keys = tuple(key.value for key in chunk_keys)
    if len(set(raw_keys)) != len(raw_keys):
        raise ValueError("duplicate requested chunk keys")
    selected_repository = (
        DjangoPrivateChunkMapRepository() if repository is None else repository
    )
    projection_ids = tuple(row.projection_id for row in scope.projections)
    located = selected_repository.locate(
        projection_ids=projection_ids, chunk_keys=raw_keys
    )
    owners: dict[str, UUID] = {}
    for projection_id, key in located:
        if key in owners:
            raise ValueError("projected chunk key has multiple selected owners")
        owners[key] = projection_id
    if set(owners) != set(raw_keys):
        raise ValueError("selected projection map does not cover graph candidates")
    by_projection = {row.projection_id: row for row in scope.projections}
    materialized = {}
    for projection_id in sorted(set(owners.values()), key=str):
        selected_keys = tuple(
            key for key in chunk_keys if owners[key.value] == projection_id
        )
        authority = by_projection[projection_id]
        for offset in range(0, len(selected_keys), _MATERIALIZATION_BATCH_SIZE):
            rows = materialize_projected_chunks(
                projection_id=projection_id,
                expected_private_mapping_checksum=authority.private_mapping_checksum,
                chunk_keys=selected_keys[offset : offset + _MATERIALIZATION_BATCH_SIZE],
                authorization=authorization,
                repository=selected_repository,
            )
            for row in rows:
                if row.chunk_key in materialized:
                    raise ValueError("duplicate materialized graph key")
                materialized[row.chunk_key] = row
    if set(materialized) != set(raw_keys):
        raise ValueError("materialized graph coverage is incomplete")
    return tuple(materialized[key] for key in raw_keys)


__all__ = [
    "DjangoPrivateChunkMapRepository",
    "materialize_selected_ready_chunks",
]
