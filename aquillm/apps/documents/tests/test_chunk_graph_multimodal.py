from __future__ import annotations

import uuid

import pytest
from django.db import connection

from ._chunk_graph_lifecycle_support import (
    configure_chunking_runtime,
    database_required,
    persist_chunk,
    persist_document,
    run_chunk_task,
)


def _embedding_values(embedding):
    return tuple(float(value) for value in embedding)


@pytest.mark.django_db(transaction=True)
@database_required
def test_duplicate_content_copy_commits_target_chunks_before_early_return_enqueue(
    monkeypatch,
):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"chunk-copy-{uuid.uuid4()}")
    source_collection = Collection.objects.create(name=f"copy source {uuid.uuid4()}")
    target_collection = Collection.objects.create(name=f"copy target {uuid.uuid4()}")
    text = "A duplicate source should copy its chunks atomically."
    source = persist_document(
        RawTextDocument,
        user=user,
        collection=source_collection,
        text=text,
        ingestion_complete=True,
    )
    target = persist_document(
        RawTextDocument,
        user=user,
        collection=target_collection,
        text=text,
        ingestion_complete=False,
    )
    source_chunk = persist_chunk(source, content=text)
    obsolete = persist_chunk(target, content="obsolete target")

    def unexpected_embedding(*_args, **_kwargs):
        raise AssertionError("duplicate-copy path must not recompute text embeddings")

    monkeypatch.setattr(chunking, "get_embeddings", unexpected_embedding)
    monkeypatch.setattr(chunking.TextChunk, "get_chunk_embedding", unexpected_embedding)
    observed = []

    def observe(document_id, source_hash):
        observed.append(
            (
                connection.in_atomic_block,
                RawTextDocument.objects.get(pkid=target.pkid).ingestion_complete,
                document_id,
                source_hash,
            )
        )

    monkeypatch.setattr(builds, "enqueue_document_build", observe)

    run_chunk_task(chunking, target)

    target.refresh_from_db()
    copied = TextChunk.objects.get(doc_id=target.id)
    assert target.ingestion_complete is True
    assert copied.pk not in {source_chunk.pk, obsolete.pk}
    assert (copied.content, copied.metadata) == (
        source_chunk.content,
        source_chunk.metadata,
    )
    assert _embedding_values(copied.embedding) == _embedding_values(
        source_chunk.embedding
    )
    assert observed == [(False, True, target.id, target.full_text_hash)]


@pytest.mark.django_db(transaction=True)
@database_required
def test_empty_duplicate_donor_falls_back_to_normal_chunk_generation(monkeypatch):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    user = User.objects.create_user(username=f"empty-donor-{uuid.uuid4()}")
    source_collection = Collection.objects.create(name=f"empty source {uuid.uuid4()}")
    target_collection = Collection.objects.create(name=f"empty target {uuid.uuid4()}")
    text = "An empty duplicate donor is not a reusable chunk snapshot."
    persist_document(
        RawTextDocument,
        user=user,
        collection=source_collection,
        text=text,
        ingestion_complete=True,
    )
    target = persist_document(
        RawTextDocument,
        user=user,
        collection=target_collection,
        text=text,
        ingestion_complete=False,
    )
    obsolete = persist_chunk(target, content="obsolete target")
    embedded_texts = []

    def embed(texts, input_type=None):
        embedded_texts.append(list(texts))
        return [[3.0] * 1024 for _text in texts]

    monkeypatch.setattr(chunking, "get_embeddings", embed)
    publications = []
    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda document_id, source_hash: publications.append(
            (document_id, source_hash)
        ),
    )

    run_chunk_task(chunking, target)

    target.refresh_from_db()
    generated = TextChunk.objects.get(doc_id=target.id)
    assert target.ingestion_complete is True
    assert generated.pk != obsolete.pk
    assert generated.content == text
    assert _embedding_values(generated.embedding) == (3.0,) * 1024
    assert embedded_texts == [[text]]
    assert publications == [(target.id, target.full_text_hash)]


@pytest.mark.django_db(transaction=True)
@database_required
@pytest.mark.parametrize(
    ("document_type_name", "source_format"),
    (("ImageUploadDocument", None), ("DocumentFigure", "pdf")),
)
def test_duplicate_text_reuses_only_text_and_embeds_target_multimodal_media(
    monkeypatch,
    document_type_name,
    source_format,
):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents import models as document_models
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.services import builds

    chunking = configure_chunking_runtime(monkeypatch)
    document_type = getattr(document_models, document_type_name)
    user = User.objects.create_user(
        username=f"copy-{document_type_name}-{uuid.uuid4()}"
    )
    source_collection = Collection.objects.create(name=f"media source {uuid.uuid4()}")
    target_collection = Collection.objects.create(name=f"media target {uuid.uuid4()}")
    text = "Duplicate OCR text does not mean duplicate image evidence."
    source_name = "source/source-only.png"
    target_name = "target/target-only.png"
    extra = {} if source_format is None else {"source_format": source_format}
    source = persist_document(
        document_type,
        user=user,
        collection=source_collection,
        text=text,
        ingestion_complete=True,
        image_file=source_name,
        **extra,
    )
    target = persist_document(
        document_type,
        user=user,
        collection=target_collection,
        text=text,
        ingestion_complete=False,
        image_file=target_name,
        **extra,
    )
    source_text = persist_chunk(source, content=text)
    source_image_content = "source-only visual evidence"
    source_image = TextChunk.objects.create(
        doc_id=source.id,
        content=source_image_content,
        start_position=len(text),
        end_position=len(text) + len(source_image_content),
        chunk_number=1,
        modality=TextChunk.Modality.IMAGE,
        metadata={"image_name": source_name},
        embedding=[1.0] * 1024,
    )
    persist_chunk(target, content="obsolete target chunk")
    embedded_images = []

    def unexpected_text_embedding(*_args, **_kwargs):
        raise AssertionError("duplicate TEXT chunks must retain saved embeddings")

    def embed_target_image(chunk, callback=None):
        assert chunk.modality == TextChunk.Modality.IMAGE
        assert chunk.metadata == {"image_name": target_name}
        embedded_images.append((chunk.content, chunk.metadata))
        chunk.embedding = [2.0] * 1024
        if callback is not None:
            callback()
        return chunk.embedding

    monkeypatch.setattr(chunking, "get_embeddings", unexpected_text_embedding)
    monkeypatch.setattr(chunking.TextChunk, "get_chunk_embedding", embed_target_image)
    publications = []
    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda document_id, source_hash: publications.append(
            (document_id, source_hash)
        ),
    )

    run_chunk_task(chunking, target)

    target.refresh_from_db()
    copied_text, target_image = TextChunk.objects.filter(doc_id=target.id).order_by(
        "chunk_number"
    )
    assert target.ingestion_complete is True
    assert copied_text.modality == TextChunk.Modality.TEXT
    assert copied_text.content == source_text.content
    assert _embedding_values(copied_text.embedding) == _embedding_values(
        source_text.embedding
    )
    assert target_image.modality == TextChunk.Modality.IMAGE
    assert target_image.content == text != source_image.content
    assert target_image.metadata == {"image_name": target_name}
    assert _embedding_values(target_image.embedding) == (2.0,) * 1024
    assert _embedding_values(target_image.embedding) != _embedding_values(
        source_image.embedding
    )
    assert embedded_images == [(text, {"image_name": target_name})]
    assert publications == [(target.id, target.full_text_hash)]
