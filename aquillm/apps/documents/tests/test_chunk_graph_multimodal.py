"""Equal text is not shared representation, media or dense-provider authority."""
from __future__ import annotations

import uuid

import pytest
from django.db import connection
from lib.embeddings.tests.test_document_provider import transport as transport

from ._chunk_graph_lifecycle_support import (
    assert_media_generation,
    assert_rebuild_binding,
    build_verified_dense,
    configure_chunking_runtime,
    database_required,
    media_storage as media_storage,
    persist_chunk,
    persist_document,
    persist_media_document,
    run_chunk_task,
)


@pytest.mark.django_db(transaction=True)
@database_required
def test_equal_content_publishes_own_generation_without_borrowing_legacy_vectors(monkeypatch):
    from django.contrib.auth.models import User
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"chunk-copy-{uuid.uuid4()}")
    source_collection = Collection.objects.create(name=f"copy source {uuid.uuid4()}")
    target_collection = Collection.objects.create(name=f"copy target {uuid.uuid4()}")
    text = "Equal canonical text still needs a target-owned generation."
    source = persist_document(RawTextDocument, user=user, collection=source_collection,
                              text=text, ingestion_complete=True)
    target = persist_document(RawTextDocument, user=user, collection=target_collection,
                              text=text, ingestion_complete=False)
    source_chunk = persist_chunk(source, content=text)
    obsolete = persist_chunk(target, content="obsolete target")

    def unexpected_embedding(*_args, **_kwargs):
        raise AssertionError("structural publication cannot call the unverified legacy embedder")
    monkeypatch.setattr(chunking, "get_embeddings", unexpected_embedding)
    monkeypatch.setattr(chunking.TextChunk, "get_chunk_embedding", unexpected_embedding)
    observed = []
    def observe(document_id, source_hash, **kwargs):
        persisted = RawTextDocument.objects.get(pkid=target.pkid)
        assert_rebuild_binding(persisted, kwargs)
        observed.append((connection.in_atomic_block, persisted.ingestion_complete,
                         tuple(persisted.chunks.values_list("pk", "content")), document_id, source_hash))
    monkeypatch.setattr(builds, "enqueue_document_build", observe)

    assert run_chunk_task(chunking, target) == "activated"
    target.refresh_from_db()
    current = target.chunks.get()
    assert target.ingestion_complete is True
    assert current.pk not in {source_chunk.pk, obsolete.pk}
    assert current.content == source_chunk.content == text
    assert current.representation_generation.owner.document_id == target.id
    assert current.embedding is None and source_chunk.embedding is not None
    assert current.metadata == {"element_keys": []} != source_chunk.metadata
    assert TextChunk.objects.get(pk=source_chunk.pk).content == text
    assert TextChunk.objects.get(pk=obsolete.pk).content == obsolete.content
    assert observed == [(False, True, ((current.pk, text),), target.id, target.full_text_hash)]


@pytest.mark.django_db(transaction=True)
@database_required
def test_empty_donor_builds_target_then_verified_dense_sidecar(monkeypatch, transport):
    from django.contrib.auth.models import User
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"empty-donor-{uuid.uuid4()}")
    source_collection = Collection.objects.create(name=f"empty source {uuid.uuid4()}")
    target_collection = Collection.objects.create(name=f"empty target {uuid.uuid4()}")
    text = "An empty duplicate donor is not a reusable chunk snapshot."
    source = persist_document(RawTextDocument, user=user, collection=source_collection,
                              text=text, ingestion_complete=True)
    target = persist_document(RawTextDocument, user=user, collection=target_collection,
                              text=text, ingestion_complete=False)
    obsolete = persist_chunk(target, content="obsolete target")
    publications = []
    monkeypatch.setattr(builds, "enqueue_document_build",
        lambda document_id, source_hash, **kw: publications.append((document_id, source_hash, kw)))

    assert run_chunk_task(chunking, target) == "activated"
    target.refresh_from_db()
    generated = target.chunks.get()
    assert target.ingestion_complete is True
    assert not TextChunk.objects.filter(doc_id=source.id).exists()
    assert generated.pk != obsolete.pk and generated.content == text
    assert generated.embedding is None and generated.search_vector is not None
    assert TextChunk.objects.get(pk=obsolete.pk).content == obsolete.content
    assert len(publications) == 1 and publications[0][:2] == (target.id, target.full_text_hash)
    assert_rebuild_binding(target, publications[0][2])

    space, calls = transport
    assert calls == []
    dense = build_verified_dense(target, space, monkeypatch)
    assert len(calls) == 1 and calls[0]["input"] == [text]
    assert tuple(float(v) for v in dense.vector) == (1.0,) * 1024
    assert dense.chunk_id == generated.pk != obsolete.pk
    generated.refresh_from_db()
    assert generated.embedding is None  # Verified dense values belong to their sidecar.
    assert target.chunks.get().pk == generated.pk


@pytest.mark.django_db(transaction=True)
@database_required
@pytest.mark.parametrize("document_type_name", ["ImageUploadDocument", "DocumentFigure"])
def test_equal_caption_retains_target_immutable_media_without_donor_reuse(
    monkeypatch, media_storage, document_type_name,
):
    from django.contrib.auth.models import User
    from apps.collections.models import Collection
    from apps.documents import models as document_models
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    document_type = getattr(document_models, document_type_name)
    user = User.objects.create_user(username=f"copy-{document_type_name}-{uuid.uuid4()}")
    source_collection = Collection.objects.create(name=f"media source {uuid.uuid4()}")
    target_collection = Collection.objects.create(name=f"media target {uuid.uuid4()}")
    text = "Duplicate OCR text does not mean duplicate image evidence."
    source_pixels, target_pixels = b"source-only immutable pixels", b"target-only immutable pixels"
    source, source_output, source_envelope = persist_media_document(document_type, user=user,
        collection=source_collection, text=text, pixels=source_pixels, monkeypatch=monkeypatch)
    assert chunking.create_chunks.run(*source_envelope[0], **source_envelope[1]) == "activated"
    source_chunk = assert_media_generation(source, source_output, source_pixels)
    target, output, envelope = persist_media_document(document_type, user=user,
        collection=target_collection, text=text, pixels=target_pixels, monkeypatch=monkeypatch)
    obsolete = persist_chunk(target, content="obsolete target chunk")
    publications = []
    monkeypatch.setattr(builds, "enqueue_document_build",
        lambda document_id, source_hash, **kw: publications.append((document_id, source_hash, kw)))

    assert chunking.create_chunks.run(*envelope[0], **envelope[1]) == "activated"
    current = assert_media_generation(target, output, target_pixels)
    assert target.ingestion_complete is True and current.content == source_chunk.content == text
    assert current.pk not in {source_chunk.pk, obsolete.pk}
    assert current.representation_generation_id != source_chunk.representation_generation_id
    assert current.representation_generation.owner.document_id == target.id
    assert output.payload["media_artifact"]["sha256"] != source_output.payload["media_artifact"]["sha256"]
    assert TextChunk.objects.get(pk=obsolete.pk).content == obsolete.content
    assert source.chunks.get().pk == source_chunk.pk
    assert len(publications) == 1 and publications[0][:2] == (target.id, target.full_text_hash)
    request = assert_rebuild_binding(target, publications[0][2])
    assert request.document_publication_state == "published"


@pytest.mark.django_db(transaction=True)
@database_required
@pytest.mark.parametrize("document_type_name", ["ImageUploadDocument", "DocumentFigure"])
def test_mutable_filename_without_durable_output_remains_unsupported(monkeypatch, document_type_name):
    from django.contrib.auth.models import User
    from apps.collections.models import Collection
    from apps.documents import models as document_models
    from apps.documents.models import TextChunk
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.knowledge_graph.models import GraphRebuildRequest
    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"unbound-media-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"unbound media {uuid.uuid4()}")
    document = persist_document(getattr(document_models, document_type_name), user=user,
        collection=collection, text="Caption without durable media", ingestion_complete=False,
        image_file="mutable-only.png")
    old = persist_chunk(document)
    with pytest.raises(SourceScopeUnavailable, match="canonical_snapshot_owner_unsupported"):
        run_chunk_task(chunking, document)
    document.refresh_from_db()
    assert document.ingestion_complete is False
    assert list(TextChunk.objects.filter(doc_id=document.id).values_list("pk", flat=True)) == [old.pk]
    assert not GraphRebuildRequest.objects.filter(scope_id=str(document.id)).exists()
