import io
from hashlib import sha256
import pytest
from pypdf import PdfReader
from .test_archive_source_binding import fake_pdf, pdf_bytes, zip_bytes


@pytest.mark.django_db
def test_worker_persists_member_pdfs_and_separate_user_authorization(
    fake_pdf, monkeypatch, settings, tmp_path
):
    from django.contrib.auth.models import User
    from django.core.files.uploadedfile import SimpleUploadedFile

    from apps.ingestion.models import SourceArtifact
    from aquillm.models import (
        Collection,
        CollectionPermission,
        IngestionBatch,
        IngestionBatchItem,
        PDFDocument,
    )
    from aquillm.task_ingest_uploaded import run_ingest_uploaded_file

    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        },
    }
    monkeypatch.setattr(
        "aquillm.models.create_chunks.delay", lambda *_args, **_kwargs: None
    )
    members = [pdf_bytes(100), pdf_bytes(200)]
    archive = zip_bytes([("a/paper.pdf", members[0]), ("b/paper.pdf", members[1])])
    owners = [User.objects.create_user(username=f"member-owner-{i}") for i in range(2)]
    collections = [Collection.objects.create(name=f"owner-{i}") for i in range(2)]
    for owner, collection in zip(owners, collections):
        CollectionPermission.objects.create(
            user=owner, collection=collection, permission="MANAGE"
        )
        item = IngestionBatchItem.objects.create(
            batch=IngestionBatch.objects.create(user=owner, collection=collection),
            source_file=SimpleUploadedFile("papers.zip", archive),
            original_filename="papers.zip",
            content_type="application/zip",
        )
        run_ingest_uploaded_file(item.id)
        item.refresh_from_db()
        assert item.status == IngestionBatchItem.Status.PARTIAL, item.error_message
        docs = list(
            PDFDocument.objects.filter(collection=collection).order_by("full_text")
        )
        assert len(docs) == 2
        for document, member in zip(docs, members):
            with document.pdf_file.open("rb") as file_obj:
                stored = file_obj.read()
            assert stored == member
            assert len(PdfReader(io.BytesIO(stored)).pages) == 1
            assert document.pdf_file.name == SourceArtifact.objects.get(sha256=sha256(member).hexdigest()).immutable_storage_key
        outputs = item.parser_metadata["outputs"]
        assert all(
            o["source_occurrence_id"] and o["source_artifact_sha256"] for o in outputs
        )
    assert (
        SourceArtifact.objects.filter(
            sha256__in=[sha256(m).hexdigest() for m in members]
        ).count()
        == 2
    )
    assert collections[0].user_can_view(owners[0])
    assert not collections[1].user_can_view(owners[0])
    assert collections[1].user_can_view(owners[1])


@pytest.mark.django_db
def test_same_title_figure_cannot_acquire_other_members_parent(
    monkeypatch, settings, tmp_path
):
    from django.contrib.auth.models import User
    from django.core.files.uploadedfile import SimpleUploadedFile

    from aquillm.ingestion.types import ExtractedTextPayload
    from aquillm.models import (
        Collection,
        CollectionPermission,
        DocumentFigure,
        IngestionBatch,
        IngestionBatchItem,
        PDFDocument,
    )
    from aquillm.task_ingest_uploaded import run_ingest_uploaded_file

    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        },
    }
    monkeypatch.setattr(
        "aquillm.models.create_chunks.delay", lambda *_args, **_kwargs: None
    )
    first, second = pdf_bytes(100), pdf_bytes(200, blank=True)

    def figures(_filename, data, _format, outputs):
        outputs.append(
            ExtractedTextPayload(
                title="paper - Figure 1",
                normalized_type="document_figure",
                full_text="first caption" if data == first else "second caption",
                modality="image",
                media_bytes=b"first image" if data == first else b"second image",
                media_filename="figure.png",
                metadata={
                    "source_document_title": "paper",
                    "source_format": "pdf",
                    "figure_index": 0,
                },
            )
        )

    monkeypatch.setattr(
        "aquillm.ingestion.parsers.extract_figure_payloads_for_format", figures
    )
    owner = User.objects.create_user(username="figure-occurrence-owner")
    collection = Collection.objects.create(name="figure occurrences")
    CollectionPermission.objects.create(user=owner, collection=collection, permission='EDIT')
    archive = zip_bytes([("a/paper.pdf", first), ("b/paper.pdf", second)])
    item = IngestionBatchItem.objects.create(
        batch=IngestionBatch.objects.create(user=owner, collection=collection),
        source_file=SimpleUploadedFile("archive.zip", archive),
        original_filename="archive.zip",
        content_type="application/zip",
    )
    run_ingest_uploaded_file(item.id)
    item.refresh_from_db()
    assert item.status == IngestionBatchItem.Status.PARTIAL, item.error_message
    source = PDFDocument.objects.get(collection=collection)
    assert (
        DocumentFigure.objects.get(full_text="first caption").parent_object_id
        == source.id
    )
    assert not DocumentFigure.objects.filter(full_text='second caption').exists()
    assert any(row['error_code'] == 'parent_output_unavailable' for row in item.parser_metadata['outputs'])
