from __future__ import annotations

import inspect
import uuid

import pytest

from ._chunk_graph_lifecycle_support import (
    assert_rebuild_binding,
    configure_chunking_runtime,
    database_required,
    persist_chunk,
    persist_document,
    run_chunk_task,
)


def test_create_chunks_task_accepts_an_exact_concrete_snapshot_reference():
    from apps.documents.tasks.chunking import create_chunks

    signature = inspect.signature(create_chunks.run)
    parameters = signature.parameters
    assert list(parameters)[:4] == [
        "doc_id",
        "expected_source_hash",
        "concrete_model_label",
        "document_pkid",
    ]
    assert list(parameters)[4:] == [
        "publication_id", "publication_generation", "publication_attempt",
    ]
    for name in list(parameters)[4:]:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None
    # Existing four-positional deliveries remain callable during rollout.
    bound = signature.bind("synthetic", "a" * 64, "apps_documents.rawtextdocument", 1)
    assert bound.arguments["document_pkid"] == 1


@pytest.mark.django_db(transaction=True)
@database_required
def test_graph_enqueue_failure_leaves_successful_chunk_commit_intact(monkeypatch):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"chunk-enqueue-fail-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"chunk enqueue failure {uuid.uuid4()}")
    document = persist_document(
        RawTextDocument,
        user=user,
        collection=collection,
        text="Chunking succeeds even when the graph broker is unavailable.",
        ingestion_complete=False,
    )
    attempted = []
    def fail(document_id, source_hash, **kwargs):
        attempted.append((document_id, source_hash, kwargs))
        raise ConnectionError("broker")
    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        fail,
    )

    run_chunk_task(chunking, document)

    document.refresh_from_db()
    assert document.ingestion_complete is True
    assert list(
        TextChunk.objects.filter(doc_id=document.id).values_list("content", flat=True)
    ) == [document.full_text]
    assert len(attempted) == 1 and attempted[0][:2] == (document.id, document.full_text_hash)
    request = assert_rebuild_binding(document, attempted[0][2])
    assert request.document_publication_state == "failed"
    assert request.error_code == "document_task_publish_failed" and request.document_publish_cursor == 0
    assert GraphRebuildRequest.objects.filter(scope_id=str(document.id)).count() == 1
    assert document.chunks.get().representation_generation.status == "active"


@pytest.mark.django_db(transaction=True)
@database_required
def test_registered_same_hash_redelivery_preserves_graph_without_duplicate_rebuild(
    monkeypatch,
):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.models import EntityMention, GraphArtifact, GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"chunk-redelivery-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"chunk redelivery {uuid.uuid4()}")
    document = persist_document(
        RawTextDocument,
        user=user,
        collection=collection,
        text="Orion evaluates MMLU.",
        ingestion_complete=True,
    )
    legacy = persist_chunk(document, content=document.full_text)
    assert run_chunk_task(chunking, document) == "activated"
    document.refresh_from_db()
    chunk = document.chunks.get()
    assert chunk.pk != legacy.pk and TextChunk.objects.filter(pk=legacy.pk).exists()
    rebuilds = tuple(GraphRebuildRequest.objects.filter(scope_id=str(document.id)).values_list("pk", flat=True))
    assert len(rebuilds) == 1

    def unexpected_provider_work(*_args, **_kwargs):
        raise AssertionError("committed same-hash redelivery must skip provider work")

    monkeypatch.setattr(chunking, "get_embeddings", unexpected_provider_work)
    monkeypatch.setattr(chunking, "get_embedding", unexpected_provider_work)
    monkeypatch.setattr(chunking, "doc_image_data_url", unexpected_provider_work)
    monkeypatch.setattr(
        chunking.TextChunk, "get_chunk_embedding", unexpected_provider_work
    )
    artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=str(document.id),
        status=GraphArtifact.Status.ACTIVE,
        source_hash=document.full_text_hash,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
    )
    mention = EntityMention.objects.create(
        artifact=artifact,
        document_id=document.id,
        chunk=chunk,
        start=0,
        end=5,
        position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
        raw_text="Orion",
        normalized_text="orion",
        entity_type="model",
        extraction_confidence=0.99,
    )
    publications = []
    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda document_id, source_hash, **kwargs: publications.append((document_id, source_hash, kwargs)),
    )

    assert run_chunk_task(chunking, document) == "already_committed"

    document.refresh_from_db()
    artifact.refresh_from_db()
    assert document.ingestion_complete is True
    assert TextChunk.objects.filter(pk=chunk.pk).exists()
    assert EntityMention.objects.filter(pk=mention.pk, chunk_id=chunk.pk).exists()
    assert artifact.status == GraphArtifact.Status.ACTIVE
    assert document.chunks.get().pk == chunk.pk
    assert TextChunk.objects.filter(pk=legacy.pk).exists()
    assert publications == []
    assert tuple(GraphRebuildRequest.objects.filter(scope_id=str(document.id)).values_list("pk", flat=True)) == rebuilds
