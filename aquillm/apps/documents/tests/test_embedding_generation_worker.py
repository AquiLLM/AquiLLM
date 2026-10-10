"""Direct worker delivery, transaction deadlines, and migration repair."""

from unittest.mock import patch
import pytest
from apps.documents.models import TextChunk
from apps.documents.services import (
    representation_store as representations,
    embedding_index_store as store,
)
from apps.documents.tests.test_embedding_generation import legacy as legacy, stage
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.embeddings.document_contracts import build_budget

pytestmark = pytest.mark.django_db(transaction=True)


def test_direct_worker_builds_vector_only_and_serializes_no_clock(legacy, transport):
    from apps.documents.tasks.embedding_index import (
        rebuild_document_embeddings,
        build_embedding_generation,
    )
    from apps.documents.models.embedding_index import EmbeddingGeneration

    space, _ = transport
    store.persist_embedding_space(space, using="default")
    selection = representations.freeze_active_representations(
        (legacy[1],), using="default"
    ).selections[0]
    before = list(TextChunk.objects.values("pk", "content", "chunk_number"))
    with patch.object(build_embedding_generation, "delay") as queue:
        key = rebuild_document_embeddings(
            selection, space_key=space.space_key, using="default"
        )
    queue.assert_called_once_with(key, using="default", timeout_ms=120000)
    assert (
        build_embedding_generation.run(key, using="default", timeout_ms=120000)
        == "activated"
    )
    assert EmbeddingGeneration.objects.get(pk=key).deliveries == 1
    assert list(TextChunk.objects.values("pk", "content", "chunk_number")) == before


def test_provider_io_runs_outside_lifecycle_transaction(legacy, transport, monkeypatch):
    from django.db import connection
    from apps.documents.tasks import embedding_index as task

    space, _ = transport
    store.persist_embedding_space(space, using="default")
    selection = representations.freeze_active_representations(
        (legacy[1],), using="default"
    ).selections[0]
    with patch.object(task.build_embedding_generation, "delay"):
        key = task.rebuild_document_embeddings(
            selection, space_key=space.space_key, using="default"
        )
    original = task.embed_document_batch

    def check(*args, **kwargs):
        assert not connection.in_atomic_block
        return original(*args, **kwargs)

    monkeypatch.setattr(task, "embed_document_batch", check)
    assert task.build_embedding_generation.run(key) == "activated"


def test_concurrent_hnsw_migration_repairs_invalid_index(legacy, transport):
    from importlib import import_module
    from django.db import connection, IntegrityError
    from apps.documents.models.embedding_index import ChunkEmbedding

    stage(legacy, transport)
    first = ChunkEmbedding.objects.get()
    second = TextChunk.objects.bulk_create(
        [
            TextChunk(
                doc_id=legacy[0].id,
                content="second",
                start_position=6,
                end_position=12,
                chunk_number=1,
            )
        ]
    )[0]
    ChunkEmbedding.objects.create(
        generation=first.generation,
        chunk=second,
        vector=first.vector,
        receipt=first.receipt,
        serving_binding=first.serving_binding,
        input_sha256=first.input_sha256,
        modality=first.modality,
        media_sha256=first.media_sha256,
        source_md5=first.source_md5,
    )
    migration = import_module(
        "apps.documents.migrations.0010_embedding_generation_hnsw_v1"
    )
    assert migration.Migration.atomic is False
    with connection.cursor() as cursor:
        cursor.execute("DROP INDEX CONCURRENTLY dense_sidecar_hnsw_v1")
        with pytest.raises(IntegrityError):
            cursor.execute(
                "CREATE UNIQUE INDEX CONCURRENTLY dense_sidecar_hnsw_v1 ON apps_documents_chunkembedding (modality)"
            )
        cursor.execute(
            "SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass('dense_sidecar_hnsw_v1')"
        )
        assert cursor.fetchone() == (False,)
    with connection.schema_editor(atomic=False) as editor:
        migration.create_index(None, editor)
        migration.create_index(None, editor)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT indisvalid, amname FROM pg_index JOIN pg_class ON pg_class.oid=indexrelid JOIN pg_am ON pg_am.oid=relam WHERE indexrelid=to_regclass('dense_sidecar_hnsw_v1')"
        )
        assert cursor.fetchone() == (True, "hnsw")


def test_build_deadline_fences_slow_activation(legacy, transport, monkeypatch):
    from apps.documents.tasks import embedding_index as task
    from apps.documents.models.embedding_index import DocumentEmbeddingState

    space, _ = transport
    store.persist_embedding_space(space, using="default")
    selection = representations.freeze_active_representations(
        (legacy[1],), using="default"
    ).selections[0]
    with patch.object(task.build_embedding_generation, "delay"):
        key = task.rebuild_document_embeddings(
            selection, space_key=space.space_key, using="default"
        )
    clock = [0.0]
    budget = build_budget(120000, clock=lambda: clock[0])
    monkeypatch.setattr(task, "build_budget", lambda timeout_ms: budget)
    original = store._manifest_current

    def slow(generation, **kwargs):
        result = original(generation, **kwargs)
        if generation.embeddings.count() == generation.expected_count:
            clock[0] = 121.0
        return result

    monkeypatch.setattr(store, "_manifest_current", slow)
    with pytest.raises(TimeoutError):
        task.build_embedding_generation.run(key)
    assert DocumentEmbeddingState.objects.get().active_generation_id is None
