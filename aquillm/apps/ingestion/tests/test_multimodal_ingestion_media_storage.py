from unittest.mock import patch

import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from aquillm.ingestion.types import ExtractedTextPayload
from aquillm.models import (
    Collection,
    DocumentFigure,
    ImageUploadDocument,
    IngestionBatch,
    IngestionBatchItem,
    MediaUploadDocument,
    RawTextDocument,
)
from aquillm.tasks import ingest_uploaded_file_task


@pytest.mark.django_db
@patch("aquillm.models.create_chunks.delay", lambda *_args, **_kwargs: None)
def test_ingest_task_dual_saves_media_and_text(monkeypatch, settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }

    user = User.objects.create_user(username="dual-save-user", password="pw12345")
    collection = Collection.objects.create(name="Dual Save Collection")
    batch = IngestionBatch.objects.create(user=user, collection=collection)
    item = IngestionBatchItem.objects.create(
        batch=batch,
        source_file=SimpleUploadedFile("mixed.zip", b"placeholder", content_type="application/zip"),
        original_filename="mixed.zip",
        content_type="application/zip",
        file_size_bytes=11,
    )

    payloads = [
        ExtractedTextPayload(
            title="whiteboard-photo",
            normalized_type="image_ocr",
            full_text="diagram text",
            modality="image",
            media_bytes=b"\x89PNG...",
            media_filename="whiteboard.png",
            media_content_type="image/png",
            provider="qwen",
            model="qwen-ocr",
        ),
        ExtractedTextPayload(
            title="meeting-audio",
            normalized_type="audio_transcript",
            full_text="meeting transcript",
            modality="audio",
            media_bytes=b"ID3...",
            media_filename="meeting.mp3",
            media_content_type="audio/mpeg",
            provider="openai",
            model="whisper-large-v3",
        ),
        ExtractedTextPayload(
            title="project-notes",
            normalized_type="txt",
            full_text="plain text notes",
            modality="text",
        ),
    ]

    monkeypatch.setattr("aquillm.ingestion.parsers.extract_text_payloads", lambda *_args, **_kwargs: payloads)

    ingest_uploaded_file_task(item.id)
    item.refresh_from_db()

    assert item.status == IngestionBatchItem.Status.SUCCESS
    assert len(item.document_ids) == 3
    assert ImageUploadDocument.objects.filter(collection=collection).count() == 1
    assert MediaUploadDocument.objects.filter(collection=collection).count() == 1
    assert RawTextDocument.objects.filter(collection=collection, title="project-notes").count() == 1

    outputs = item.parser_metadata.get("outputs", [])
    assert len(outputs) == 3
    assert any(output.get("modality") == "image" and output.get("raw_media_saved") for output in outputs)
    assert any(output.get("modality") == "audio" and output.get("raw_media_saved") for output in outputs)
    assert any(output.get("modality") == "text" and output.get("text_extracted") for output in outputs)


@pytest.mark.django_db
@patch("aquillm.models.create_chunks.delay", lambda *_args, **_kwargs: None)
def test_ingest_task_allows_multiple_no_text_figures(monkeypatch, settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }

    user = User.objects.create_user(username="figure-user", password="pw12345")
    collection = Collection.objects.create(name="Figure Collection")
    batch = IngestionBatch.objects.create(user=user, collection=collection)
    item = IngestionBatchItem.objects.create(
        batch=batch,
        source_file=SimpleUploadedFile("slides.pptx", b"pptx", content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        original_filename="slides.pptx",
        content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        file_size_bytes=4,
    )

    payloads = [
        ExtractedTextPayload(
            title="slides",
            normalized_type="pptx",
            full_text="slide deck text",
            modality="text",
        ),
        ExtractedTextPayload(
            title="slide-1-figure",
            normalized_type="document_figure",
            full_text="",
            modality="image",
            media_bytes=b"\x89PNGfigure1",
            media_filename="slide1.png",
            media_content_type="image/png",
            metadata={
                "source_format": "pptx",
                "source_document_title": "slides",
                "figure_index": 0,
                "location_metadata": {"slide_number": 1},
            },
        ),
        ExtractedTextPayload(
            title="slide-2-figure",
            normalized_type="document_figure",
            full_text="",
            modality="image",
            media_bytes=b"\x89PNGfigure2",
            media_filename="slide2.png",
            media_content_type="image/png",
            metadata={
                "source_format": "pptx",
                "source_document_title": "slides",
                "figure_index": 1,
                "location_metadata": {"slide_number": 2},
            },
        ),
    ]

    monkeypatch.setattr("aquillm.ingestion.parsers.extract_text_payloads", lambda *_args, **_kwargs: payloads)

    ingest_uploaded_file_task(item.id)
    item.refresh_from_db()

    assert item.status == IngestionBatchItem.Status.SUCCESS
    source_doc = RawTextDocument.objects.get(collection=collection, title="slides")
    assert DocumentFigure.objects.filter(collection=collection).count() == 0

    figures = list(DocumentFigure.objects.all())
    assert len(figures) == 2
    assert len({figure.collection_id for figure in figures}) == 1
    assert figures[0].collection.parent_id == collection.id
    assert figures[0].collection.name.endswith(" - Figures")
    assert all(figure.parent_object_id == source_doc.id for figure in figures)


@pytest.mark.django_db
@patch("aquillm.models.create_chunks.delay", lambda *_args, **_kwargs: None)
def test_ingest_task_allows_duplicate_figure_text(monkeypatch, settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }

    user = User.objects.create_user(username="figure-text-user", password="pw12345")
    collection = Collection.objects.create(name="Figure Text Collision Collection")
    batch = IngestionBatch.objects.create(user=user, collection=collection)
    item = IngestionBatchItem.objects.create(
        batch=batch,
        source_file=SimpleUploadedFile("slides.pptx", b"pptx", content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        original_filename="slides.pptx",
        content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        file_size_bytes=4,
    )

    payloads = [
        ExtractedTextPayload(
            title="slides",
            normalized_type="pptx",
            full_text="repeated slides text",
            modality="text",
        ),
        ExtractedTextPayload(
            title="slide-1-figure",
            normalized_type="document_figure",
            full_text="Repeated extracted caption",
            modality="image",
            media_bytes=b"\x89PNGfigureA",
            media_filename="slide1.png",
            media_content_type="image/png",
            metadata={
                "source_format": "pptx",
                "source_document_title": "slides",
                "figure_index": 0,
                "location_metadata": {"slide_number": 1},
            },
        ),
        ExtractedTextPayload(
            title="slide-2-figure",
            normalized_type="document_figure",
            full_text="Repeated extracted caption",
            modality="image",
            media_bytes=b"\x89PNGfigureB",
            media_filename="slide2.png",
            media_content_type="image/png",
            metadata={
                "source_format": "pptx",
                "source_document_title": "slides",
                "figure_index": 1,
                "location_metadata": {"slide_number": 2},
            },
        ),
    ]

    monkeypatch.setattr("aquillm.ingestion.parsers.extract_text_payloads", lambda *_args, **_kwargs: payloads)

    ingest_uploaded_file_task(item.id)
    item.refresh_from_db()

    assert item.status == IngestionBatchItem.Status.SUCCESS
    source_doc = RawTextDocument.objects.get(collection=collection, title="slides")
    assert DocumentFigure.objects.filter(collection=collection).count() == 0

    figures = list(DocumentFigure.objects.all())
    assert len(figures) == 2
    assert len({figure.collection_id for figure in figures}) == 1
    assert figures[0].collection.parent_id == collection.id
    assert figures[0].collection.name.endswith(" - Figures")
    assert all(figure.parent_object_id == source_doc.id for figure in figures)
