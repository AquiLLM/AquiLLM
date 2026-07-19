"""Private PDF responses stream from both S3-shaped and filesystem storage."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from botocore.exceptions import ClientError
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse, StreamingHttpResponse
from django.urls import reverse
from django.utils.http import content_disposition_header, http_date
from storages.backends.s3 import S3Storage

from aquillm.models import (
    Collection,
    CollectionPermission,
    PDFDocument,
    RawTextDocument,
)

PDF_STREAM_CHUNK_SIZE = 64 * 1024

pytestmark = pytest.mark.django_db


class CountingBody:
    """Small SDK-body stand-in that records bounded reads and close calls."""

    def __init__(self, content: bytes):
        self.content = content
        self.offset = 0
        self.read_sizes: list[int] = []
        self.close_count = 0

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            size = len(self.content) - self.offset
        chunk = self.content[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.close_count += 1


class FakeS3Object:
    def __init__(self, *, response=None, error: ClientError | None = None):
        self.response = response
        self.error = error
        self.get_count = 0

    def get(self):
        self.get_count += 1
        if self.error is not None:
            raise self.error
        return self.response


class FakeS3Bucket:
    def __init__(self, object_: FakeS3Object):
        self.object = object_
        self.requested_keys: list[str] = []

    def Object(self, key: str) -> FakeS3Object:  # noqa: N802 - boto3 SDK API
        self.requested_keys.append(key)
        return self.object


class FakeS3Storage(S3Storage):
    """S3Storage-compatible fake without constructing a boto3 connection."""

    def __init__(self, bucket: FakeS3Bucket):
        self._fake_bucket = bucket
        self.names_to_normalize: list[str] = []

    @property
    def bucket(self) -> FakeS3Bucket:
        return self._fake_bucket

    def _normalize_name(self, name: str) -> str:
        self.names_to_normalize.append(name)
        return f"private/{name}"


@pytest.fixture
def authorized_client(client):
    user = User.objects.create_user(username="pdf-viewer")
    collection = Collection.objects.create(name="PDF responses")
    CollectionPermission.objects.create(
        user=user,
        collection=collection,
        permission="VIEW",
    )
    client.force_login(user)
    return client, user, collection


def _pdf_document(*, user, collection, name: str) -> PDFDocument:
    full_text = "non-empty extracted text"
    doc = PDFDocument(
        title="Quarterly report",
        full_text=full_text,
        full_text_hash=PDFDocument.hash_fn(full_text),
        collection=collection,
        ingested_by=user,
        pdf_file=name,
    )
    doc.save(skip_text_extraction=True, dont_rechunk=True)
    return doc


def _raw_text_document(*, user, collection, name: str) -> RawTextDocument:
    full_text = "non-empty crawled text"
    doc = RawTextDocument(
        title="Captured page",
        full_text=full_text,
        full_text_hash=RawTextDocument.hash_fn(full_text),
        collection=collection,
        ingested_by=user,
        rendered_pdf=name,
    )
    doc.save(dont_rechunk=True)
    return doc


def _client_error(status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": str(status), "Message": "storage error"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "GetObject",
    )


def _install_pdf_storage(monkeypatch, storage) -> None:
    monkeypatch.setattr(PDFDocument._meta.get_field("pdf_file"), "storage", storage)


def test_s3_pdf_is_incremental_and_forwards_private_response_headers(
    authorized_client,
    monkeypatch,
):
    client, user, collection = authorized_client
    content = b"%PDF-1.7\n" + (b"a" * (PDF_STREAM_CHUNK_SIZE + 417))
    body = CountingBody(content)
    last_modified = datetime(2026, 7, 1, 12, 30, tzinfo=UTC)
    object_ = FakeS3Object(
        response={
            "Body": body,
            "ContentLength": len(content),
            "ETag": '"etag-value"',
            "LastModified": last_modified,
        }
    )
    bucket = FakeS3Bucket(object_)
    storage = FakeS3Storage(bucket)
    _install_pdf_storage(monkeypatch, storage)
    name = "reports/drafts/../r\u00e9sum\u00e9 \"Q3\".pdf"
    clean_storage_name = 'reports/r\u00e9sum\u00e9 "Q3".pdf'
    doc = _pdf_document(user=user, collection=collection, name=name)

    response = client.get(reverse("pdf", kwargs={"doc_id": doc.id}))

    assert isinstance(response, StreamingHttpResponse)
    assert response["Content-Type"] == "application/pdf"
    assert response["Content-Length"] == str(len(content))
    assert response["ETag"] == '"etag-value"'
    assert response["Last-Modified"] == http_date(last_modified.timestamp())
    assert response["Content-Disposition"] == content_disposition_header(
        False,
        'r\u00e9sum\u00e9 "Q3".pdf',
    )
    assert response["Cache-Control"] == "private, max-age=300"
    assert storage.names_to_normalize == [clean_storage_name]
    assert bucket.requested_keys == [f"private/{clean_storage_name}"]

    chunks = iter(response.streaming_content)
    first = next(chunks)
    assert first == content[:PDF_STREAM_CHUNK_SIZE]
    assert body.read_sizes == [PDF_STREAM_CHUNK_SIZE]
    assert first + b"".join(chunks) == content
    assert body.close_count == 1
    response.close()
    assert body.close_count == 1


def test_s3_response_close_before_consumption_closes_body(
    authorized_client,
    monkeypatch,
):
    client, user, collection = authorized_client
    content = b"%PDF-1.7\nnot-yet-consumed"
    body = CountingBody(content)
    object_ = FakeS3Object(response={"Body": body, "ContentLength": len(content)})
    storage = FakeS3Storage(FakeS3Bucket(object_))
    _install_pdf_storage(monkeypatch, storage)
    doc = _pdf_document(user=user, collection=collection, name="pdfs/early-close.pdf")

    response = client.get(reverse("pdf", kwargs={"doc_id": doc.id}))
    response.close()

    assert body.read_sizes == []
    assert body.close_count == 1


def test_s3_missing_key_returns_404(authorized_client, monkeypatch):
    client, user, collection = authorized_client
    object_ = FakeS3Object(error=_client_error(404))
    _install_pdf_storage(monkeypatch, FakeS3Storage(FakeS3Bucket(object_)))
    doc = _pdf_document(user=user, collection=collection, name="pdfs/missing.pdf")

    response = client.get(reverse("pdf", kwargs={"doc_id": doc.id}))

    assert response.status_code == 404
    response.close()


def test_s3_zero_length_closes_body_and_returns_404(authorized_client, monkeypatch):
    client, user, collection = authorized_client
    body = CountingBody(b"")
    object_ = FakeS3Object(response={"Body": body, "ContentLength": 0})
    _install_pdf_storage(monkeypatch, FakeS3Storage(FakeS3Bucket(object_)))
    doc = _pdf_document(user=user, collection=collection, name="pdfs/empty.pdf")

    response = client.get(reverse("pdf", kwargs={"doc_id": doc.id}))

    assert response.status_code == 404
    assert body.read_sizes == []
    assert body.close_count == 1
    response.close()


def test_s3_unrelated_client_error_propagates(authorized_client, monkeypatch):
    client, user, collection = authorized_client
    error = _client_error(403)
    object_ = FakeS3Object(error=error)
    _install_pdf_storage(monkeypatch, FakeS3Storage(FakeS3Bucket(object_)))
    doc = _pdf_document(user=user, collection=collection, name="pdfs/forbidden.pdf")

    with pytest.raises(ClientError) as exc_info:
        client.get(reverse("pdf", kwargs={"doc_id": doc.id}))

    assert exc_info.value is error


def test_filesystem_pdf_streams_body_and_private_headers(
    authorized_client,
    monkeypatch,
    tmp_path,
):
    client, user, collection = authorized_client
    storage = FileSystemStorage(location=tmp_path)
    _install_pdf_storage(monkeypatch, storage)
    content = b"%PDF-1.7\nfilesystem-payload"
    name = storage.save("pdfs/local report.pdf", ContentFile(content))
    doc = _pdf_document(user=user, collection=collection, name=name)

    response = client.get(reverse("pdf", kwargs={"doc_id": doc.id}))

    try:
        assert isinstance(response, FileResponse)
        assert response["Content-Type"] == "application/pdf"
        assert response["Content-Length"] == str(len(content))
        assert response["Content-Disposition"] == content_disposition_header(
            False,
            "local report.pdf",
        )
        assert response["Cache-Control"] == "private, max-age=300"
        assert b"".join(response.streaming_content) == content
    finally:
        response.close()


def test_filesystem_pdf_without_a_file_returns_404(
    authorized_client,
    monkeypatch,
    tmp_path,
):
    client, user, collection = authorized_client
    _install_pdf_storage(monkeypatch, FileSystemStorage(location=tmp_path))
    doc = _pdf_document(user=user, collection=collection, name="")

    response = client.get(reverse("pdf", kwargs={"doc_id": doc.id}))

    assert response.status_code == 404
    response.close()


def test_filesystem_missing_pdf_returns_404(authorized_client, monkeypatch, tmp_path):
    client, user, collection = authorized_client
    _install_pdf_storage(monkeypatch, FileSystemStorage(location=tmp_path))
    doc = _pdf_document(user=user, collection=collection, name="pdfs/not-there.pdf")

    response = client.get(reverse("pdf", kwargs={"doc_id": doc.id}))

    assert response.status_code == 404
    response.close()


def test_filesystem_empty_pdf_returns_404(authorized_client, monkeypatch, tmp_path):
    client, user, collection = authorized_client
    storage = FileSystemStorage(location=tmp_path)
    _install_pdf_storage(monkeypatch, storage)
    name = storage.save("pdfs/empty.pdf", ContentFile(b""))
    doc = _pdf_document(user=user, collection=collection, name=name)

    response = client.get(reverse("pdf", kwargs={"doc_id": doc.id}))

    assert response.status_code == 404
    response.close()


def test_filesystem_pdf_remains_permission_protected(client, monkeypatch, tmp_path):
    viewer = User.objects.create_user(username="unauthorized-pdf-viewer")
    owner = User.objects.create_user(username="pdf-owner")
    collection = Collection.objects.create(name="Private PDF collection")
    storage = FileSystemStorage(location=tmp_path)
    _install_pdf_storage(monkeypatch, storage)
    name = storage.save("pdfs/private.pdf", ContentFile(b"%PDF-private"))
    doc = _pdf_document(user=owner, collection=collection, name=name)
    client.force_login(viewer)

    response = client.get(reverse("pdf", kwargs={"doc_id": doc.id}))

    assert response.status_code == 403
    response.close()


def test_raw_text_rendered_pdf_uses_filesystem_fallback(
    authorized_client,
    monkeypatch,
    tmp_path,
):
    client, user, collection = authorized_client
    storage = FileSystemStorage(location=tmp_path)
    monkeypatch.setattr(
        RawTextDocument._meta.get_field("rendered_pdf"),
        "storage",
        storage,
    )
    content = b"%PDF-1.7\nrendered-web-page"
    name = storage.save("crawled_pdfs/capture.pdf", ContentFile(content))
    doc = _raw_text_document(user=user, collection=collection, name=name)

    response = client.get(reverse("pdf", kwargs={"doc_id": doc.id}))

    try:
        assert isinstance(response, FileResponse)
        assert response["Content-Type"] == "application/pdf"
        assert response["Content-Disposition"] == content_disposition_header(
            False,
            "capture.pdf",
        )
        assert response["Cache-Control"] == "private, max-age=300"
        assert b"".join(response.streaming_content) == content
    finally:
        response.close()
