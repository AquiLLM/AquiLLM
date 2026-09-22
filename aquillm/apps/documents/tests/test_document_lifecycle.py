"""Non-graph document lifecycle and atomic chunk publication regressions."""

import pytest
from django.contrib.auth.models import User
from apps.collections.models import Collection
from apps.documents.models import DocumentFigure, RawTextDocument, TextChunk
from apps.documents.tasks.chunking import _commit_chunks, create_chunks


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
    monkeypatch.setattr(create_chunks, "delay", lambda *args: calls.append(args))
    with django_capture_on_commit_callbacks(execute=True):
        document.full_text = "updated text"
        document.save(update_fields=["full_text"])
        assert not calls
    document.refresh_from_db()
    assert calls == [
        (
            str(document.id),
            document.hash_fn("updated text"),
            "apps_documents.rawtextdocument",
            document.pkid,
        )
    ]
    assert not document.ingestion_complete


def test_stale_chunk_publication_preserves_current_evidence(document):
    TextChunk.objects.bulk_create([chunk_for(document)])
    stale_hash = document.full_text_hash
    document.full_text = "changed source"
    document.save(dont_rechunk=True)
    assert (
        _commit_chunks(
            document,
            [chunk_for(document, "obsolete")],
            expected_source_hash=stale_hash,
            using="default",
        )
        == "stale"
    )
    assert list(document.chunks.values_list("content", flat=True)) == ["old evidence"]


def test_chunk_publication_is_atomic_and_idempotent(document):
    TextChunk.objects.bulk_create([chunk_for(document)])
    assert (
        _commit_chunks(
            document,
            [chunk_for(document, "new evidence")],
            expected_source_hash=document.full_text_hash,
            using="default",
        )
        == "committed"
    )
    assert (
        _commit_chunks(
            document,
            [chunk_for(document, "duplicate job")],
            expected_source_hash=document.full_text_hash,
            using="default",
        )
        == "already_committed"
    )
    assert list(document.chunks.values_list("content", flat=True)) == ["new evidence"]


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
