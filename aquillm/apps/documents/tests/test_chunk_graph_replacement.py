from __future__ import annotations

import uuid

import pytest
from django.db import connection, transaction

from ._chunk_graph_lifecycle_support import (
    configure_chunking_runtime,
    database_required,
    persist_chunk,
    persist_document,
    run_chunk_task,
)


@pytest.mark.django_db(transaction=True)
@database_required
def test_normal_image_chunking_atomically_replaces_text_and_image_chunks(monkeypatch):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import ImageUploadDocument, TextChunk
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"chunk-image-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"chunk image {uuid.uuid4()}")
    document = persist_document(
        ImageUploadDocument,
        user=user,
        collection=collection,
        text="Orion evaluates MMLU.",
        ingestion_complete=False,
        image_file="ingestion_images/orion.png",
    )
    obsolete = persist_chunk(document)
    publications = []
    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda document_id, source_hash: publications.append((document_id, source_hash)),
    )

    run_chunk_task(chunking, document)

    document.refresh_from_db()
    chunks = list(TextChunk.objects.filter(doc_id=document.id).order_by("chunk_number"))
    assert document.ingestion_complete is True
    assert not TextChunk.objects.filter(pk=obsolete.pk).exists()
    assert [chunk.chunk_number for chunk in chunks] == [0, 1]
    assert [chunk.modality for chunk in chunks] == ["text", "image"]
    assert chunks[0].content == document.full_text
    assert chunks[1].metadata["image_name"] == "ingestion_images/orion.png"
    assert publications == [(document.id, document.full_text_hash)]


@pytest.mark.django_db(transaction=True)
@database_required
def test_graph_publication_observes_only_the_committed_text_replacement(monkeypatch):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"chunk-order-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"chunk order {uuid.uuid4()}")
    document = persist_document(
        RawTextDocument,
        user=user,
        collection=collection,
        text="The replacement chunk is committed before publication.",
        ingestion_complete=False,
    )
    obsolete = persist_chunk(document)
    observed = []

    def observe(document_id, source_hash):
        persisted = RawTextDocument.objects.get(pkid=document.pkid, id=document_id)
        observed.append(
            (
                connection.in_atomic_block,
                persisted.ingestion_complete,
                tuple(
                    TextChunk.objects.filter(doc_id=document_id)
                    .order_by("chunk_number")
                    .values_list("pk", "content")
                ),
                document_id,
                source_hash,
            )
        )

    monkeypatch.setattr(builds, "enqueue_document_build", observe)

    run_chunk_task(chunking, document)

    replacement = TextChunk.objects.get(doc_id=document.id)
    assert observed == [
        (
            False,
            True,
            ((replacement.pk, document.full_text),),
            document.id,
            document.full_text_hash,
        )
    ]
    assert replacement.pk != obsolete.pk


@pytest.mark.django_db(transaction=True)
@database_required
def test_stale_expected_hash_does_not_mutate_chunks_or_publish_graph_work(monkeypatch):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"chunk-stale-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"chunk stale {uuid.uuid4()}")
    document = persist_document(
        RawTextDocument,
        user=user,
        collection=collection,
        text="This is the current document body.",
        ingestion_complete=False,
    )
    original = persist_chunk(document, content="current preserved chunk")
    before = tuple(
        TextChunk.objects.filter(doc_id=document.id).values_list(
            "pk", "content", "chunk_number", "metadata"
        )
    )
    publications = []
    monkeypatch.setattr(
        builds, "enqueue_document_build", lambda *args: publications.append(args)
    )

    run_chunk_task(chunking, document, expected_source_hash="f" * 64)

    document.refresh_from_db()
    after = tuple(
        TextChunk.objects.filter(doc_id=document.id).values_list(
            "pk", "content", "chunk_number", "metadata"
        )
    )
    assert after == before
    assert after[0][0] == original.pk
    assert document.ingestion_complete is False
    assert publications == []


@pytest.mark.django_db(transaction=True)
@database_required
def test_outer_rollback_discards_replacement_chunks_and_graph_publication(monkeypatch):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"chunk-rollback-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"chunk rollback {uuid.uuid4()}")
    document = persist_document(
        RawTextDocument,
        user=user,
        collection=collection,
        text="The task writes inside the caller's transaction boundary.",
        ingestion_complete=False,
    )
    original = persist_chunk(document, content="rollback preserves this chunk")
    publications = []
    monkeypatch.setattr(
        builds, "enqueue_document_build", lambda *args: publications.append(args)
    )

    with pytest.raises(RuntimeError, match="force outer rollback"):
        with transaction.atomic():
            run_chunk_task(chunking, document)
            assert publications == []
            raise RuntimeError("force outer rollback")

    document.refresh_from_db()
    assert document.ingestion_complete is False
    assert list(
        TextChunk.objects.filter(doc_id=document.id).values_list("pk", "content")
    ) == [(original.pk, original.content)]
    assert publications == []
