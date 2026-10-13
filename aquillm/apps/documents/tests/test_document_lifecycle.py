"""Non-graph document lifecycle and atomic chunk publication regressions."""

from uuid import UUID

import pytest
from django.contrib.auth.models import User

from apps.collections.models import Collection
from apps.documents.models import DocumentFigure, DocumentRepresentationState, RawTextDocument, TextChunk
from apps.documents.models.chunk_publication import ChunkPublication
from apps.documents.tasks.chunking import create_chunks
from .representation_support import retained as retained, upload as upload
from .test_representation_tasks import delivery


@pytest.fixture
def document(db):
    user = User.objects.create(username="lifecycle-user")
    collection = Collection.objects.create(name="Lifecycle")
    return RawTextDocument.objects.create(
        title="Source",
        full_text="original text",
        collection=collection,
        ingested_by=user,
    )


def chunk_for(document, content="old evidence"):
    return TextChunk(
        doc_id=document.id,
        content=content,
        start_position=0,
        end_position=len(content),
        chunk_number=0,
    )


def test_content_save_publishes_exact_identity_only_after_commit(
    document, monkeypatch, django_capture_on_commit_callbacks
):
    calls = []
    monkeypatch.setattr(
        create_chunks, "delay", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    with django_capture_on_commit_callbacks(execute=True):
        document.full_text = "updated text"
        document.save(update_fields=["full_text"])
        assert not calls
    document.refresh_from_db()
    intent = ChunkPublication.objects.get(
        concrete_model_label="apps_documents.rawtextdocument",
        document_pkid=document.pkid,
        document_id=document.id,
        source_hash=document.hash_fn("updated text"),
    )
    assert intent.pk > 0
    assert isinstance(intent.generation, UUID)
    assert intent.attempts == 1
    assert calls == [
        (
            (
                str(document.id),
                document.hash_fn("updated text"),
                "apps_documents.rawtextdocument",
                document.pkid,
            ),
            {
                "publication_id": intent.pk,
                "publication_generation": str(intent.generation),
                "publication_attempt": intent.attempts,
            },
        )
    ]
    assert not document.ingestion_complete


@pytest.mark.django_db(transaction=True)
def test_stale_chunk_publication_preserves_current_evidence(retained):
    document, _, old = retained
    envelope = delivery(document)
    assert create_chunks.run(**envelope) == "activated"
    document.refresh_from_db()
    state = DocumentRepresentationState.objects.get(document_id=document.id)
    predecessor = state.active_generation_id
    history = list(TextChunk.objects.filter(doc_id=document.id).order_by("pk").values_list("pk", "content"))
    document.full_text = "changed source"
    document.save(dont_rechunk=True)
    assert not document.chunks.exists()
    assert create_chunks.run(**envelope) == "stale"
    state.refresh_from_db()
    assert state.active_generation_id == predecessor
    assert not document.chunks.exists()
    assert list(TextChunk.objects.filter(doc_id=document.id).order_by("pk").values_list("pk", "content")) == history
    assert TextChunk.objects.get(pk=old.pk).content == old.content


@pytest.mark.django_db(transaction=True)
def test_chunk_publication_is_atomic_and_idempotent(retained):
    document, output, old = retained
    envelope = delivery(document)
    assert create_chunks.run(**envelope) == "activated"
    document.refresh_from_db()
    state = DocumentRepresentationState.objects.get(document_id=document.id)
    generation = state.active_generation_id
    current = list(document.chunks.values_list("pk", "content"))
    assert current and old.pk not in {pk for pk, _ in current}
    assert state.active_generation.extraction_revision_id == output.extraction_revision_id
    assert create_chunks.run(**envelope) == "stale"
    assert create_chunks.run(str(document.id), document.full_text_hash,
        document._meta.label_lower, document.pkid) == "already_committed"
    state.refresh_from_db()
    assert state.active_generation_id == generation and state.generations.count() == 1
    assert list(document.chunks.values_list("pk", "content")) == current
    assert TextChunk.objects.get(pk=old.pk).content == old.content


@pytest.mark.parametrize(
    "model_name",
    [
        "RawTextDocument",
        "PDFDocument",
        "TeXDocument",
        "VTTDocument",
        "HandwrittenNotesDocument",
        "ImageUploadDocument",
        "MediaUploadDocument",
    ],
)
def test_queryset_parent_deletion_cleans_figure_and_parent_chunks(
    document, model_name, monkeypatch
):
    from apps.documents import models

    model = getattr(models, model_name)
    if model_name == "HandwrittenNotesDocument":
        monkeypatch.setattr(model, "extract_text", lambda self: None)
    parent = model.objects.create(
        title="Parent",
        full_text="parent text",
        collection=document.collection,
        ingested_by=document.ingested_by,
    )
    figure = DocumentFigure(
        title="Figure",
        full_text="figure text",
        collection=parent.collection,
        ingested_by=parent.ingested_by,
        image_file="figure.png",
        parent_document=parent,
    )
    figure.save(dont_rechunk=True)
    ids = [parent.id, figure.id]
    TextChunk.objects.bulk_create([chunk_for(parent), chunk_for(figure)])
    model.objects.filter(pk=parent.pk).delete()
    assert not DocumentFigure.objects.filter(pk=figure.pk).exists()
    assert not TextChunk.objects.filter(doc_id__in=ids).exists()
