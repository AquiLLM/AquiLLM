from __future__ import annotations

import uuid

import pytest
from django.db import connection, transaction

from ._chunk_graph_lifecycle_support import (
    assert_media_generation,
    assert_rebuild_binding,
    configure_chunking_runtime,
    database_required,
    media_storage as media_storage,
    persist_media_document,
    persist_chunk,
    persist_document,
    run_chunk_task,
)


@pytest.mark.django_db(transaction=True)
@database_required
def test_durable_image_chunking_publishes_caption_and_retains_history(monkeypatch, media_storage):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import ImageUploadDocument, TextChunk
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"chunk-image-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"chunk image {uuid.uuid4()}")
    pixels = b"target immutable image pixels"
    document, output, envelope = persist_media_document(
        ImageUploadDocument,
        user=user,
        collection=collection,
        text="Orion evaluates MMLU.",
        pixels=pixels, monkeypatch=monkeypatch,
    )
    obsolete = persist_chunk(document)
    publications = []
    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda document_id, source_hash, **kwargs: publications.append((document_id, source_hash, kwargs)),
    )

    wrong = dict(envelope[1], publication_attempt=envelope[1]["publication_attempt"] + 1)
    assert chunking.create_chunks.run(*envelope[0], **wrong) == "stale"
    assert publications == []
    assert list(TextChunk.objects.filter(doc_id=document.id).values_list("pk", flat=True)) == [obsolete.pk]
    assert chunking.create_chunks.run(*envelope[0], **envelope[1]) == "activated"

    document.refresh_from_db()
    current = assert_media_generation(document, output, pixels)
    assert document.ingestion_complete is True
    assert TextChunk.objects.get(pk=obsolete.pk).content == obsolete.content
    assert current.pk != obsolete.pk and current.chunk_number == 0
    assert len(publications) == 1 and publications[0][:2] == (document.id, document.full_text_hash)
    request = assert_rebuild_binding(document, publications[0][2])
    assert request.document_publication_state == "published"
    assert chunking.create_chunks.run(*envelope[0], **envelope[1]) == "stale"
    assert len(publications) == 1 and document.chunks.get().pk == current.pk


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

    bindings = []
    def observe(document_id, source_hash, **kwargs):
        persisted = RawTextDocument.objects.get(pkid=document.pkid, id=document_id)
        bindings.append(assert_rebuild_binding(persisted, kwargs))
        observed.append(
            (
                connection.in_atomic_block,
                persisted.ingestion_complete,
                tuple(
                    persisted.chunks
                    .order_by("chunk_number")
                    .values_list("pk", "content")
                ),
                document_id,
                source_hash,
            )
        )

    monkeypatch.setattr(builds, "enqueue_document_build", observe)

    run_chunk_task(chunking, document)

    document.refresh_from_db()
    replacement = document.chunks.get()
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
    assert TextChunk.objects.get(pk=obsolete.pk).content == obsolete.content
    assert replacement.representation_generation_id is not None
    assert len(bindings) == 1


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
        builds, "enqueue_document_build", lambda *args, **kwargs: publications.append((args, kwargs))
    )

    assert run_chunk_task(chunking, document, expected_source_hash="f" * 64) == "stale"

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
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.documents.models import DocumentRepresentationGeneration

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
        builds, "enqueue_document_build", lambda *args, **kwargs: publications.append((args, kwargs))
    )

    with pytest.raises(RuntimeError, match="force outer rollback"):
        with transaction.atomic():
            assert run_chunk_task(chunking, document) == "activated"
            document.refresh_from_db()
            assert document.chunks.get().pk != original.pk
            assert GraphRebuildRequest.objects.filter(scope_id=str(document.id)).count() == 1
            assert publications == []
            raise RuntimeError("force outer rollback")

    document.refresh_from_db()
    assert document.ingestion_complete is False
    assert list(
        TextChunk.objects.filter(doc_id=document.id).values_list("pk", "content")
    ) == [(original.pk, original.content)]
    assert publications == []
    assert not GraphRebuildRequest.objects.filter(scope_id=str(document.id)).exists()
    assert not DocumentRepresentationGeneration.objects.filter(owner__document_id=document.id).exists()
