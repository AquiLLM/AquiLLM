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
        self.fenced: tuple[uuid.UUID, str, int] | None = None

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

    def fence(self, *, projection_id, checksum, row_count):
        self.fenced = (projection_id, checksum, row_count)


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
    assert store.fenced == (projection_id, first, 1)


def test_chunk_reference_persistence_fails_closed_for_partial_or_stale_map() -> None:
    class PartialStore(_ChunkStore):
        def create(self, **_kwargs):
            self.created += 1

    partial = PartialStore()
    repository = PostgresProjectionRepository(source=object(), chunk_store=partial)
    with pytest.raises(ValueError, match="incomplete"):
        repository.persist_chunk_references(
            projection_id=uuid.uuid4(), rows=(_row(),), batch_size=10
        )
    assert partial.fenced is None

    stale = _ChunkStore()
    requested = _row()
    stale.create(projection_id=uuid.uuid4(), rows=(requested,), batch_size=10)
    extra = PrivateProjectionChunkReferenceV1("f" * 64, 102, str(uuid.UUID(int=3)), 4)
    stale.create(projection_id=uuid.uuid4(), rows=(extra,), batch_size=10)
    repository = PostgresProjectionRepository(source=object(), chunk_store=stale)
    with pytest.raises(ValueError, match="incomplete"):
        repository.persist_chunk_references(
            projection_id=uuid.uuid4(), rows=(requested,), batch_size=10
        )
    assert stale.fenced is None


def test_chunk_reference_persistence_rejects_a_stale_current_chunk() -> None:
    store = _ChunkStore()
    repository = PostgresProjectionRepository(source=object(), chunk_store=store)
    row = _row()
    projection_id = uuid.uuid4()
    repository.persist_chunk_references(
        projection_id=projection_id, rows=(row,), batch_size=10
    )
    store.fenced = None
    store.rows[row.projection_chunk_key].chunk_id = None

    with pytest.raises(ValueError, match="stale"):
        repository.persist_chunk_references(
            projection_id=projection_id, rows=(row,), batch_size=10
        )

    assert store.fenced is None


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
