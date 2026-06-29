"""Citation API: chunk_detail modality/image_url and the citation_sources batch."""
from __future__ import annotations

import json

import pytest
from django.contrib.auth.models import User
from django.urls import reverse

from aquillm.models import Collection, CollectionPermission, RawTextDocument, TextChunk


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
def test_chunk_detail_text_chunk_has_text_modality_and_no_image_url(client):
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
    # source_url now flows from the base Document model.
    assert data["document"]["source_url"] == "https://arxiv.org/abs/1234.5678"


@pytest.mark.django_db
def test_chunk_detail_image_chunk_exposes_image_url(client):
    user = User.objects.create_user(username="cd-img", password="pw12345")
    collection = Collection.objects.create(name="CD Img")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    doc = RawTextDocument.objects.create(
        title="Figure doc", full_text="caption text", collection=collection, ingested_by=user,
    )
    chunk = _make_chunk(doc, content="a figure caption", modality=TextChunk.Modality.IMAGE)
    assert client.login(username="cd-img", password="pw12345")

    resp = client.get(reverse("api_chunk_detail", kwargs={"chunk_id": chunk.pk}))
    assert resp.status_code == 200
    data = resp.json()
    assert data["modality"] == "image"
    assert data["image_url"] == f"/aquillm/document_image/{doc.id}/"


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
