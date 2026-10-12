"""The chunk lookup migration retains restricted execution through a roundtrip."""
import json
import os
from uuid import uuid4

import pytest
from django.db import DatabaseError, connection, transaction
from django.db.migrations.executor import MigrationExecutor
from django.utils import timezone

from apps.collections.models import Collection
from apps.documents.models import TextChunk
from apps.knowledge_graph.models import ProjectionChunkReference
from apps.knowledge_graph.tests.test_projection_locking_postgres import (
    _active_artifact,
    _building_projection,
)

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="requires disposable PostgreSQL state-role functions",
    ),
]


def _function_identity():
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT p.oid,p.proowner,p.proacl::text,p.prosecdef,p.proconfig,p.prosrc, "
            "has_function_privilege('aquillm_projection_state',p.oid,'EXECUTE'), "
            "EXISTS (SELECT 1 FROM aclexplode(COALESCE(p.proacl,acldefault('f',p.proowner))) a "
            "WHERE a.grantee=0 AND a.privilege_type='EXECUTE') "
            "FROM pg_proc p WHERE p.oid="
            "'public.kg_projection_store_chunk_references(uuid,text,jsonb,timestamptz)'::regprocedure"
        )
        row = cursor.fetchone()
    assert row is not None
    assert row[3] is True
    assert row[4] == ["search_path=pg_catalog, public"]
    assert row[6:] == (True, False)
    return row


def _store(projection, rows, owner="race-worker"):
    with transaction.atomic(), connection.cursor() as cursor:
        cursor.execute("SET LOCAL ROLE aquillm_projection_state")
        cursor.execute(
            "SELECT stored_count FROM public.kg_projection_store_chunk_references(%s,%s,%s,%s)",
            [projection.pk, owner, json.dumps(rows), timezone.now()],
        )
        return cursor.fetchall()


def _exercise_restricted_store():
    collection = Collection.objects.create(name="chunk migration fixture")
    artifact = _active_artifact(collection)
    _, projection = _building_projection(collection, artifact, timezone.now())
    document = uuid4()
    chunks = TextChunk.objects.bulk_create([
        TextChunk(content="fixture", doc_id=document, chunk_number=index,
                  start_position=index * 7, end_position=index * 7 + 7)
        for index in range(2)
    ])
    rows = [dict(projection_chunk_key=f"{index:064x}", integer_chunk_pk=chunk.pk,
                 document_uuid=str(document), chunk_number=index)
            for index, chunk in enumerate(chunks)]
    # Duplicate inputs remain valid; unique storage inserts once and replay inserts zero.
    assert _store(projection, [rows[0], rows[0]]) == [(1,)]
    assert _store(projection, [rows[0], rows[0]]) == [(0,)]
    assert _store(projection, [rows[1]], owner="not-the-owner") == []
    for change in ({"document_uuid": str(uuid4())}, {"chunk_number": 99}):
        invalid = dict(rows[1], **change)
        with pytest.raises(DatabaseError, match="invalid projection chunk coordinate"):
            _store(projection, [rows[1], invalid])
        assert list(ProjectionChunkReference.objects.filter(projection=projection)
                    .values_list("integer_chunk_pk", "document_uuid", "chunk_number")) == [
                        (chunks[0].pk, document, 0)]
    assert _store(projection, [rows[1]]) == [(1,)]
    assert list(ProjectionChunkReference.objects.filter(projection=projection)
                .order_by("chunk_number")
                .values_list("integer_chunk_pk", "document_uuid", "chunk_number")) == [
                    (chunks[0].pk, document, 0), (chunks[1].pk, document, 1)]


def test_chunk_lookup_migration_roundtrip_preserves_acl_and_atomic_coordinates(monkeypatch):
    assert connection.vendor == "postgresql"
    monkeypatch.setenv("KG_PROJECTION_IDENTIFIER_HMAC_KEY", "audit-test-key-only")
    monkeypatch.setenv("KG_PROJECTION_IDENTIFIER_KEY_VERSION", "key-v1")
    latest = MigrationExecutor(connection).loader.graph.leaf_nodes()
    before = _function_identity()
    try:
        _exercise_restricted_store()
        MigrationExecutor(connection).migrate([
            ("apps_knowledge_graph", "0014_projection_representation_boundary_v1")])
        prior = _function_identity()
        assert prior[:5] == before[:5]
        assert prior[5] != before[5]
        _exercise_restricted_store()
        MigrationExecutor(connection).migrate(latest)
        assert _function_identity() == before
        _exercise_restricted_store()
    finally:
        MigrationExecutor(connection).migrate(latest)
