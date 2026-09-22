"""Complete multi-page source/mapping reads against real PostgreSQL functions."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from django.db import ProgrammingError, connection, transaction
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from apps.collections.models import Collection
from apps.documents.models import TextChunk
from apps.knowledge_graph.projection.chunk_reference_store import (
    DjangoChunkReferenceStore,
)
from apps.knowledge_graph.projection.django_projection_rows import _bounded
from apps.knowledge_graph.projection.postgres_repository import (
    PostgresProjectionRepository,
)
from apps.knowledge_graph.projection.records import PrivateProjectionChunkReferenceV1
from apps.knowledge_graph.projection.serialization import private_chunk_mapping_checksum
from apps.knowledge_graph.projection.state_repository import (
    FunctionProjectionStateRepository,
)
from apps.knowledge_graph.tests.test_projection_locking_postgres import (
    _active_artifact,
    _building_projection,
)

pytestmark = [
    pytest.mark.django_db,
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="set KG_REQUIRE_POSTGRES_TESTS=1 for PostgreSQL integration tests",
    ),
]


@pytest.fixture
def bulk_mapping(monkeypatch):
    assert connection.vendor == "postgresql"
    monkeypatch.setenv("KG_PROJECTION_IDENTIFIER_HMAC_KEY", "audit-test-key-only")
    monkeypatch.setenv("KG_PROJECTION_IDENTIFIER_KEY_VERSION", "key-v1")
    collection = Collection.objects.create(name=f"bulk projection audit {uuid4()}")
    artifact = _active_artifact(collection)
    _, projection = _building_projection(collection, artifact, timezone.now())
    document = uuid4()
    chunks = TextChunk.objects.bulk_create(
        [
            TextChunk(
                content="fixture",
                doc_id=document,
                chunk_number=index,
                start_position=index * 7,
                end_position=index * 7 + 7,
            )
            for index in range(10_005)
        ],
        batch_size=500,
    )
    values = tuple(
        PrivateProjectionChunkReferenceV1(
            f"{index:064x}", chunk.pk, str(document), index
        )
        for index, chunk in enumerate(chunks)
    )
    store = FunctionProjectionStateRepository(
        state_using="default", source_using="default", owner="race-worker"
    )
    return projection, document, values, store


def test_complete_sql_source_and_chunk_mapping_have_exact_checksum(bulk_mapping):
    projection, document, values, store = bulk_mapping
    loaded = _bounded(
        TextChunk.objects.filter(doc_id=document),
        ("id", "doc_id", "chunk_number"),
        500,
    )
    assert len(loaded) == 10_005
    assert loaded[-1]["chunk_number"] == 10_004
    assert all(set(row) == {"id", "doc_id", "chunk_number"} for row in loaded)
    repository = PostgresProjectionRepository(source=object(), chunk_store=store)
    checksum = repository.persist_chunk_references(
        projection_id=projection.pk, rows=values, batch_size=500
    )
    assert checksum == private_chunk_mapping_checksum(values)
    with CaptureQueriesContext(connection) as captured:
        private_rows = store.load(projection_id=projection.pk)
        assert len(private_rows) == len(values)
        for row in private_rows:
            assert row.chunk.id == row.integer_chunk_pk
            assert row.chunk.doc_id == row.document_uuid
            assert row.chunk.chunk_number == row.chunk_number
            assert {"content", "embedding"}.issubset(row.chunk.get_deferred_fields())
    assert len(captured) == 1
    with CaptureQueriesContext(connection) as captured:
        DjangoChunkReferenceStore("default").fence(
            projection_id=projection.pk, checksum=checksum, row_count=len(values)
        )
    # Coordinate validation must not issue one query per chunk.
    assert len(captured) < 8
    assert (
        repository.persist_chunk_references(
            projection_id=projection.pk, rows=values, batch_size=499
        )
        == checksum
    )
    projection.refresh_from_db()
    assert projection.private_mapping_checksum == checksum
    assert checksum != private_chunk_mapping_checksum(values[:-1])


def test_bulk_mapping_state_role_and_source_lease_fences_remain_exact(bulk_mapping):
    projection, document, values, store = bulk_mapping
    store.create(projection_id=projection.pk, rows=values, batch_size=500)
    checksum = private_chunk_mapping_checksum(values)
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL ROLE aquillm_projection_state")
        store.fence(
            projection_id=projection.pk, checksum=checksum, row_count=len(values)
        )
        with pytest.raises(ProgrammingError, match="permission denied"):
            with transaction.atomic(), connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE public.apps_knowledge_graph_collectiongraphprojection "
                    "SET private_mapping_checksum = %s WHERE id = %s",
                    ["f" * 64, projection.pk],
                )
        with connection.cursor() as cursor:
            cursor.execute("RESET ROLE")
    store.owner = "different-worker"
    with pytest.raises(ValueError, match="fence was lost"):
        store.fence(
            projection_id=projection.pk, checksum=checksum, row_count=len(values)
        )
    store.owner = "race-worker"
    with pytest.raises(ValueError, match="fence was lost"):
        store.fence(
            projection_id=projection.pk, checksum=checksum, row_count=len(values) - 1
        )
    # A changed source coordinate must fail even after all write pages succeeded.
    TextChunk.objects.filter(pk=values[-1].integer_chunk_pk).update(chunk_number=20_000)
    with pytest.raises(ValueError, match="fence was lost"):
        store.fence(
            projection_id=projection.pk, checksum=checksum, row_count=len(values)
        )
