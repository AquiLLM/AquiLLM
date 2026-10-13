"""Actual owner deletion between citation chunk selection and exact lookup."""
import json
from unittest.mock import Mock

import pytest
from django.test import RequestFactory

from apps.chat.models import Message, WSConversation
from apps.documents.models import DocumentRepresentationState, TextChunk
from apps.documents.views import api
from .representation_support import activate, retained, upload

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("route", ["citation_narrow", "citation_sources"])
@pytest.mark.parametrize("binding", ["generation", "legacy_snapshot"])
def test_deleted_exact_owner_is_unavailable_without_server_error(
    retained, monkeypatch, route, binding,
):
    doc, output, old = retained
    assert activate(doc, output) == "activated"
    doc.refresh_from_db()
    old.refresh_from_db()
    chunk = doc.chunks.get() if binding == "generation" else old
    assert chunk.representation_generation_id if binding == "generation" else chunk.legacy_source_snapshot_id
    chunk_id, owner_pk, document_id = chunk.pk, doc.pk, doc.id
    user = doc.ingested_by
    conversation = WSConversation.objects.create(owner=user, name="citation race")
    message = Message.objects.create(conversation=conversation, role="assistant",
        content=f"Saved claim [doc:{document_id} chunk:{chunk_id}].", sequence_number=0)
    provider = Mock(return_value="must not be requested")
    monkeypatch.setattr(api, "narrow_citation", provider)
    original = api._get_chunk_document
    selected = []

    def remove_owner_then_lookup(selected_chunk, **kwargs):
        assert selected_chunk.pk == chunk_id
        assert original(selected_chunk, **kwargs).pk == owner_pk
        selected.append(selected_chunk.pk)
        # The API has already materialized this real chunk. Delete its actual
        # owner through the normal lifecycle, then delegate the stale object's
        # lookup to the unchanged exact-owner service; no exception is mocked.
        doc.delete()
        assert not type(doc).objects.filter(pk=owner_pk).exists()
        assert not TextChunk.objects.filter(pk=chunk_id).exists()
        assert not DocumentRepresentationState.objects.filter(
            model_label=doc._meta.label_lower, document_pkid=owner_pk,
            document_id=document_id).exists()
        return original(selected_chunk, **kwargs)

    monkeypatch.setattr(api, "_get_chunk_document", remove_owner_then_lookup)
    body = ({"chunk_id": chunk_id, "message_uuid": str(message.message_uuid)}
            if route == "citation_narrow" else {"chunk_ids": [chunk_id]})
    request = RequestFactory().post("/citation/", data=json.dumps(body),
                                    content_type="application/json")
    request.user = user
    response = getattr(api, route)(request)
    assert selected == [chunk_id]
    provider.assert_not_called()
    if route == "citation_narrow":
        assert response.status_code == 404
        assert json.loads(response.content) == {"error": "Chunk's document not found"}
    else:
        assert response.status_code == 200
        assert json.loads(response.content) == {"sources": []}
