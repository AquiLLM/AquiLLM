from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    OpaqueProjectionKey,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.projection.postgres_repository import (
    PostgresProjectionRepository,
)
from apps.knowledge_graph.projection.records import PrivateProjectionChunkReferenceV1
from apps.knowledge_graph.projection.serialization import private_chunk_mapping_checksum


class _ChunkStore:
    def __init__(self) -> None:
        self.rows: dict[str, object] = {}
        self.created = 0

    def load(self, *, projection_id, keys=None):
        values = self.rows.values()
        if keys is not None:
            values = (row for row in values if row.projection_chunk_key in keys)
        return tuple(values)

    def create(self, *, projection_id, rows, batch_size):
        for row in rows:
            self.created += 1
            self.rows[row.projection_chunk_key] = SimpleNamespace(
                projection_chunk_key=row.projection_chunk_key,
                integer_chunk_pk=row.integer_chunk_pk,
                document_uuid=uuid.UUID(row.document_uuid),
                chunk_number=row.chunk_number,
                chunk_id=row.integer_chunk_pk,
                chunk=SimpleNamespace(
                    pk=row.integer_chunk_pk,
                    doc_id=uuid.UUID(row.document_uuid),
                    chunk_number=row.chunk_number,
                ),
            )


def _row() -> PrivateProjectionChunkReferenceV1:
    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1")
    key = codec.encode(
        ProjectionIdentifierDomain.CHUNK, generation=uuid.UUID(int=1), source=101
    )
    return PrivateProjectionChunkReferenceV1(str(key), 101, str(uuid.UUID(int=2)), 3)


def _opaque(row: PrivateProjectionChunkReferenceV1) -> OpaqueProjectionKey:
    return OpaqueProjectionKey(
        ProjectionIdentifierDomain.CHUNK, row.projection_chunk_key
    )


def test_chunk_reference_persistence_is_idempotent_and_checksum_exact() -> None:
    store = _ChunkStore()
    repository = PostgresProjectionRepository(source=object(), chunk_store=store)
    row = _row()
    projection_id = uuid.uuid4()

    first = repository.persist_chunk_references(
        projection_id=projection_id, rows=(row,), batch_size=10
    )
    second = repository.persist_chunk_references(
        projection_id=projection_id, rows=(row,), batch_size=10
    )

    assert first == second == private_chunk_mapping_checksum((row,))
    assert store.created == 1


def test_chunk_reference_resolution_rejects_deleted_stale_and_conflicting_rows() -> (
    None
):
    store = _ChunkStore()
    repository = PostgresProjectionRepository(source=object(), chunk_store=store)
    row = _row()
    projection_id = uuid.uuid4()
    repository.persist_chunk_references(
        projection_id=projection_id, rows=(row,), batch_size=10
    )
    stored = store.rows[row.projection_chunk_key]

    stored.chunk_id = None
    with pytest.raises(ValueError, match="stale"):
        repository.resolve_projection_chunk_references(
            projection_id=projection_id,
            chunk_keys=(_opaque(row),),
            authorized_document_ids=frozenset({uuid.UUID(row.document_uuid)}),
        )
    stored.chunk_id = row.integer_chunk_pk
    stored.chunk.chunk_number = 99
    with pytest.raises(ValueError, match="conflict"):
        repository.resolve_projection_chunk_references(
            projection_id=projection_id,
            chunk_keys=(_opaque(row),),
            authorized_document_ids=frozenset({uuid.UUID(row.document_uuid)}),
        )


def test_chunk_reference_resolution_requires_current_authorized_document() -> None:
    store = _ChunkStore()
    repository = PostgresProjectionRepository(source=object(), chunk_store=store)
    row = _row()
    projection_id = uuid.uuid4()
    repository.persist_chunk_references(
        projection_id=projection_id, rows=(row,), batch_size=1
    )

    assert (
        repository.resolve_projection_chunk_references(
            projection_id=projection_id,
            chunk_keys=(_opaque(row),),
            authorized_document_ids=frozenset(),
        )
        == ()
    )
