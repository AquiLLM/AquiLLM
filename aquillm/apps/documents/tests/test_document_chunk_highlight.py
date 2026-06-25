"""Document page ?chunk=… highlights a TextChunk when it belongs to the document."""
from __future__ import annotations

import pytest
from django.contrib.auth.models import User
from django.urls import reverse

from aquillm.models import Collection, CollectionPermission, RawTextDocument, TextChunk


@pytest.mark.django_db
def test_document_page_shows_chunk_when_chunk_query_matches(client, settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
    user = User.objects.create_user(username="chunk-highlight-user", password="pw12345")
    collection = Collection.objects.create(name="Chunk Highlight Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    doc = RawTextDocument.objects.create(
        title="Paper",
        full_text="full body",
        collection=collection,
        ingested_by=user,
    )
    chunk = TextChunk.objects.create(
        content="excerpt inside chunk",
        start_position=0,
        end_position=20,
        chunk_number=0,
        doc_id=doc.id,
        embedding=[0.0] * 1024,
    )
    assert client.login(username="chunk-highlight-user", password="pw12345")
    url = reverse("document", kwargs={"doc_id": doc.id})
    response = client.get(url, {"chunk": str(chunk.pk)})
    assert response.status_code == 200
    assert b"excerpt inside chunk" in response.content
    assert b"Cited excerpt" in response.content


@pytest.mark.django_db
def test_document_page_ignores_chunk_that_belongs_to_another_document(client, settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
    user = User.objects.create_user(username="chunk-mismatch-user", password="pw12345")
    collection = Collection.objects.create(name="Mismatch Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    doc_a = RawTextDocument.objects.create(
        title="Doc A",
        full_text="aaa",
        collection=collection,
        ingested_by=user,
    )
    doc_b = RawTextDocument.objects.create(
        title="Doc B",
        full_text="bbb",
        collection=collection,
        ingested_by=user,
    )
    chunk = TextChunk.objects.create(
        content="only on doc a",
        start_position=0,
        end_position=10,
        chunk_number=0,
        doc_id=doc_a.id,
        embedding=[0.0] * 1024,
    )
    assert client.login(username="chunk-mismatch-user", password="pw12345")
    url_b = reverse("document", kwargs={"doc_id": doc_b.id})
    response = client.get(url_b, {"chunk": str(chunk.pk)})
    assert response.status_code == 200
    assert b"only on doc a" not in response.content
