from __future__ import annotations

import inspect
import uuid

import pytest

from ._chunk_graph_lifecycle_support import (
    configure_chunking_runtime,
    database_required,
    persist_chunk,
    persist_document,
    run_chunk_task,
)


def test_create_chunks_task_accepts_an_exact_concrete_snapshot_reference():
    from apps.documents.tasks.chunking import create_chunks

    assert list(inspect.signature(create_chunks.run).parameters) == [
        "doc_id",
        "expected_source_hash",
        "concrete_model_label",
        "document_pkid",
    ]


@pytest.mark.django_db(transaction=True)
@database_required
def test_graph_enqueue_failure_leaves_successful_chunk_commit_intact(monkeypatch):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.graph import invalidation
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
    events = []
    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionError("broker")),
    )
    monkeypatch.setattr(
        invalidation.logger,
        "error",
        lambda event, **fields: events.append((event, fields)),
    )

    run_chunk_task(chunking, document)

    document.refresh_from_db()
    assert document.ingestion_complete is True
    assert list(
        TextChunk.objects.filter(doc_id=document.id).values_list("content", flat=True)
    ) == [document.full_text]
    assert events == [
        (
            "obs.kg.document_enqueue_failed",
            {
                "document_id": str(document.id),
                "expected_source_hash": document.full_text_hash,
                "error_type": "ConnectionError",
            },
        )
    ]


@pytest.mark.django_db(transaction=True)
@database_required
def test_same_hash_redelivery_preserves_active_graph_evidence_and_reenqueues(
    monkeypatch,
):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.models import EntityMention, GraphArtifact
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
    chunk = persist_chunk(document, content=document.full_text)

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
        lambda document_id, source_hash: publications.append((document_id, source_hash)),
    )

    run_chunk_task(chunking, document)

    document.refresh_from_db()
    artifact.refresh_from_db()
    assert document.ingestion_complete is True
    assert TextChunk.objects.filter(pk=chunk.pk).exists()
    assert EntityMention.objects.filter(pk=mention.pk, chunk_id=chunk.pk).exists()
    assert artifact.status == GraphArtifact.Status.ACTIVE
    assert publications == [(document.id, document.full_text_hash)]
