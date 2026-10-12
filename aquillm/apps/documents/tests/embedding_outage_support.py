"""Real lexical publication and dense delivery; only external transport is synthetic."""
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from django.utils import timezone

from apps.documents.models import TextChunk, DocumentRepresentationState
from apps.documents.models.embedding_index import EmbeddingGeneration
from apps.documents.services import embedding_index_store as store
from apps.documents.services.representation_history import reference
from apps.documents.services.representation_snapshot_build import build_canonical_snapshot_representation
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.tasks import embedding_index as task
from apps.documents.tests.test_embedding_generation import legacy as legacy
from lib.embeddings import document_provider
from lib.embeddings.tests.test_document_provider import transport as transport


@pytest.fixture
def dense_source(legacy, transport, monkeypatch):
    from apps.documents.tasks import chunking
    document, _ = legacy
    monkeypatch.setattr("apps.knowledge_graph.services.builds.enqueue_document_build", lambda *a, **k: None)
    assert build_canonical_snapshot_representation(reference(document), using="default") == "activated"
    document.refresh_from_db()
    frozen = freeze_active_representations((reference(document),), using="default")
    selection = frozen.selections[0]
    assert selection.generation_key is not None
    store.persist_embedding_space(transport[0], using="default")
    forbidden = Mock(side_effect=AssertionError("dense worker used legacy per-chunk fallback"))
    monkeypatch.setattr(chunking, "get_embeddings", forbidden)
    monkeypatch.setattr(chunking, "get_embedding", forbidden)
    monkeypatch.setattr(TextChunk, "get_chunk_embedding", forbidden)
    source = document, selection, transport[0]
    before = source_state(source)
    yield source
    forbidden.assert_not_called()
    document.refresh_from_db()
    if document.full_text_hash == selection.document.source_hash:
        from apps.documents.services.representation_store import validate_representation_snapshot
        validate_representation_snapshot(frozen, using="default")
        assert source_state(source) == before


def schedule(source):
    with patch.object(task.build_embedding_generation, "delay") as queue:
        key = task.rebuild_document_embeddings(source[1], space_key=source[2].space_key, using="default")
    queue.assert_called_once_with(key, using="default", timeout_ms=120000)
    return key


def due(key):
    assert EmbeddingGeneration.objects.filter(pk=key).update(
        next_attempt_at=timezone.now() - timedelta(seconds=1)) == 1


def transport_failure(monkeypatch, failure):
    """Run the actual preparation/adapter/error classifier around this SDK edge."""
    original = document_provider._client
    dispatch = Mock(side_effect=failure)
    monkeypatch.setattr(document_provider, "_client",
        lambda: SimpleNamespace(embeddings=SimpleNamespace(create=dispatch)))
    return dispatch, original


def source_state(source):
    document = source[0]
    return (
        DocumentRepresentationState.objects.get(document_id=document.id).active_generation_id,
        list(TextChunk.objects.filter(doc_id=document.id).order_by("pk").values(
            "pk", "content", "chunk_number", "start_position", "end_position",
            "representation_generation_id", "legacy_source_snapshot_id", "search_vector_binding")),
    )


def assert_dense_failed(key, kind, *, deliveries=1):
    row = EmbeddingGeneration.objects.select_related("owner").get(pk=key)
    assert (row.state, row.failure_kind, row.deliveries) == ("failed", kind, deliveries)
    assert row.owner.active_generation_id is None
    assert not row.embeddings.exists()
    return row


def assert_dense_ready(key, source):
    row = EmbeddingGeneration.objects.select_related("owner").get(pk=key)
    assert row.state == "ready" and not row.failure_kind
    assert row.owner.active_generation_id == row.key
    assert row.owner.representation_key == source[1].generation_key
    assert row.embeddings.count() == row.expected_count > 0
    assert set(row.embeddings.values_list("chunk_id", flat=True)) == {
        item["chunk_id"] for item in row.manifest}


def current_lease(key):
    from lib.embeddings.document_contracts import EmbeddingLeaseV1
    row = EmbeddingGeneration.objects.get(pk=key)
    return EmbeddingLeaseV1(row.key, row.lease_token)
