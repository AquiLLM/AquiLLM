"""Citation API: chunk_detail modality/image_url and the citation_sources batch."""
from __future__ import annotations

import json

import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from aquillm.models import (
    Collection,
    CollectionPermission,
    PDFDocument,
    RawTextDocument,
    TextChunk,
)


def _make_chunk(doc, *, content="excerpt", number=0, modality=TextChunk.Modality.TEXT):
    # Positions must be unique per (doc, start, end); offset by chunk number.
    start = number * 1000
    return TextChunk.objects.create(
        content=content,
        start_position=start,
        end_position=start + len(content),
        chunk_number=number,
        doc_id=doc.id,
        modality=modality,
        embedding=[0.0] * 1024,
    )


@pytest.mark.django_db
def test_chunk_detail_default_text_includes_full_text_and_text_offset(client):
    user = User.objects.create_user(username="cd-text", password="pw12345")
    collection = Collection.objects.create(name="CD Text")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    doc = RawTextDocument.objects.create(
        title="Paper", full_text="full body text", collection=collection, ingested_by=user,
        source_url="https://arxiv.org/abs/1234.5678",
    )
    chunk = _make_chunk(doc, content="full body")
    assert client.login(username="cd-text", password="pw12345")

    resp = client.get(reverse("api_chunk_detail", kwargs={"chunk_id": chunk.pk}))
    assert resp.status_code == 200
    data = resp.json()
    assert data["modality"] == "text"
    assert data["image_url"] is None
    assert data["document"]["full_text"] == "full body text"
    assert data["document"]["text_offset"] == 0
    # source_url now flows from the base Document model.
    assert data["document"]["source_url"] == "https://arxiv.org/abs/1234.5678"


@pytest.mark.django_db
def test_chunk_detail_compact_text_keeps_metadata_without_full_text(client):
    user = User.objects.create_user(username="cd-compact-text", password="pw12345")
    collection = Collection.objects.create(name="CD Compact Text")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    doc = RawTextDocument.objects.create(
        title="Compact paper",
        full_text="full compact body",
        collection=collection,
        ingested_by=user,
        source_url="https://example.com/compact-paper",
    )
    chunk = _make_chunk(doc, content="compact excerpt", number=2)
    assert client.login(username="cd-compact-text", password="pw12345")

    with CaptureQueriesContext(connection) as queries:
        resp = client.get(
            reverse("api_chunk_detail", kwargs={"chunk_id": chunk.pk}),
            {"include_full_text": "0"},
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "content": "compact excerpt",
        "chunk_number": 2,
        "start_position": 2000,
        "end_position": 2015,
        "start_time": None,
        "modality": "text",
        "image_url": None,
        "document": {
            "id": str(doc.id),
            "title": "Compact paper",
            "type": "RawTextDocument",
            "has_pdf": False,
            "source_url": "https://example.com/compact-paper",
        },
    }
    document_selects = [
        query["sql"]
        for query in queries.captured_queries
        if 'FROM "aquillm_rawtextdocument"' in query["sql"]
    ]
    assert len(document_selects) == 1
    assert '"aquillm_rawtextdocument"."full_text"' not in document_selects[0]


@pytest.mark.django_db
def test_chunk_detail_compact_image_chunk_exposes_image_url(client):
    user = User.objects.create_user(username="cd-img", password="pw12345")
    collection = Collection.objects.create(name="CD Img")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    doc = RawTextDocument.objects.create(
        title="Figure doc", full_text="caption text", collection=collection, ingested_by=user,
        source_url="https://example.com/figure-source",
    )
    chunk = _make_chunk(doc, content="a figure caption", modality=TextChunk.Modality.IMAGE)
    assert client.login(username="cd-img", password="pw12345")

    resp = client.get(
        reverse("api_chunk_detail", kwargs={"chunk_id": chunk.pk}),
        {"include_full_text": "0"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["modality"] == "image"
    assert data["image_url"] == f"/aquillm/document_image/{doc.id}/"
    assert data["start_position"] == 0
    assert data["end_position"] == len("a figure caption")
    assert data["document"] == {
        "id": str(doc.id),
        "title": "Figure doc",
        "type": "RawTextDocument",
        "has_pdf": False,
        "source_url": "https://example.com/figure-source",
    }


@pytest.mark.django_db
def test_chunk_detail_compact_pdf_keeps_metadata_without_full_text(
    client, settings, tmp_path
):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        },
    }
    user = User.objects.create_user(username="cd-pdf", password="pw12345")
    collection = Collection.objects.create(name="CD PDF")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    doc = PDFDocument.objects.create(
        title="PDF paper",
        full_text="non-empty extracted PDF text",
        collection=collection,
        ingested_by=user,
        source_url="https://example.com/pdf-source",
        pdf_file=SimpleUploadedFile(
            "paper.pdf", b"not parsed by save", content_type="application/pdf"
        ),
    )
    chunk = _make_chunk(doc, content="PDF excerpt", number=3)
    assert client.login(username="cd-pdf", password="pw12345")

    resp = client.get(
        reverse("api_chunk_detail", kwargs={"chunk_id": chunk.pk}),
        {"include_full_text": "0"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "PDF excerpt"
    assert data["chunk_number"] == 3
    assert data["start_position"] == 3000
    assert data["end_position"] == 3011
    assert data["start_time"] is None
    assert data["modality"] == "text"
    assert data["image_url"] is None
    assert data["document"] == {
        "id": str(doc.id),
        "title": "PDF paper",
        "type": "PDFDocument",
        "has_pdf": True,
        "source_url": "https://example.com/pdf-source",
    }


@pytest.mark.django_db
@pytest.mark.parametrize("query", [{}, {"include_full_text": "0"}])
def test_chunk_detail_default_and_compact_preserve_403(client, query):
    user = User.objects.create_user(username="cd-denied", password="pw12345")
    owner = User.objects.create_user(username="cd-owner", password="pw12345")
    collection = Collection.objects.create(name="CD Denied")
    CollectionPermission.objects.create(user=owner, collection=collection, permission="VIEW")
    doc = RawTextDocument.objects.create(
        title="Private paper",
        full_text="private body",
        collection=collection,
        ingested_by=owner,
    )
    chunk = _make_chunk(doc)
    assert client.login(username="cd-denied", password="pw12345")

    resp = client.get(
        reverse("api_chunk_detail", kwargs={"chunk_id": chunk.pk}), query
    )

    assert resp.status_code == 403


@pytest.mark.django_db
@pytest.mark.parametrize("query", [{}, {"include_full_text": "0"}])
def test_chunk_detail_default_and_compact_preserve_404(client, query):
    user = User.objects.create_user(username="cd-missing", password="pw12345")
    assert client.login(username="cd-missing", password="pw12345")

    resp = client.get(
        reverse("api_chunk_detail", kwargs={"chunk_id": 99999999}), query
    )

    assert resp.status_code == 404


@pytest.mark.django_db
def test_citation_sources_groups_and_enforces_access(client):
    user = User.objects.create_user(username="cs-user", password="pw12345")
    other = User.objects.create_user(username="cs-other", password="pw12345")

    visible = Collection.objects.create(name="Visible")
    CollectionPermission.objects.create(user=user, collection=visible, permission="VIEW")
    hidden = Collection.objects.create(name="Hidden")
    CollectionPermission.objects.create(user=other, collection=hidden, permission="VIEW")

    doc = RawTextDocument.objects.create(
        title="Visible Paper", full_text="x", collection=visible, ingested_by=user,
    )
    secret = RawTextDocument.objects.create(
        title="Secret Paper", full_text="y", collection=hidden, ingested_by=other,
    )
    c1 = _make_chunk(doc, content="one", number=0)
    c2 = _make_chunk(doc, content="two", number=1, modality=TextChunk.Modality.IMAGE)
    c_secret = _make_chunk(secret, content="hidden", number=0)

    assert client.login(username="cs-user", password="pw12345")
    resp = client.post(
        reverse("api_citation_sources"),
        data=json.dumps({"chunk_ids": [c1.pk, c2.pk, c_secret.pk, 99999999]}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    sources = resp.json()["sources"]

    returned_ids = {s["chunk_id"] for s in sources}
    # Visible chunks returned; the inaccessible chunk and the missing id dropped.
    assert returned_ids == {c1.pk, c2.pk}
    by_id = {s["chunk_id"]: s for s in sources}
    assert by_id[c1.pk]["doc_id"] == str(doc.id)
    assert by_id[c1.pk]["title"] == "Visible Paper"
    assert by_id[c2.pk]["modality"] == "image"


@pytest.mark.django_db
def test_citation_sources_rejects_non_list(client):
    user = User.objects.create_user(username="cs-bad", password="pw12345")
    assert client.login(username="cs-bad", password="pw12345")
    resp = client.post(
        reverse("api_citation_sources"),
        data=json.dumps({"chunk_ids": "not-a-list"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
