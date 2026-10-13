"""A persisted old answer retains its exact source after a real parser activation."""
import pytest
from channels.db import database_sync_to_async
from django.test import Client
from django.urls import reverse

from apps.chat.services.rag_turn import preservation_turn
from apps.collections.models import CollectionPermission
from apps.documents.models import TextChunk
from apps.documents.tests.representation_support import (
    activate, retained as retained, upload as upload,
)
from aquillm.message_adapters import load_conversation_from_db
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.types.messages import UserMessage
from .cross_lane_publication_support import (
    Provider, QUESTION, REVISED, SOURCE, cited_reply, consumer_for, publish, retrieve,
    close_publication_worker as close_publication_worker,
    publication_source as publication_source,
)

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "source"])
async def test_persisted_citation_survives_new_parser_and_obeys_current_view(publication_source, monkeypatch, mode):
    source = publication_source
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", mode)
    old = source.chunk
    citation = f"[doc:{source.doc.id} chunk:{old.pk}]"
    provider = Provider([cited_reply(SOURCE, citation)])
    consumer = consumer_for(source)
    async with preservation_turn(consumer, max_func_calls=3):
        message = await retrieve(consumer, provider)
        assert citation in message.content
        assert "α😀 Capacity is 18 litres at 20 C." in message.content
        assert "The capacity applies only below 1 Pa." in message.content
        assert message._validated_answer.outcome != "limited"
        saved, _ = await publish(source, consumer)
    assert len(provider.requests) == 1 and not provider.answers
    assert SOURCE in str(provider.requests[0]["messages"])
    original_uuid = message.message_uuid
    loaded = await database_sync_to_async(load_conversation_from_db)(source.db)
    assert loaded[-1].content == saved and loaded[-1]._validated_answer is None
    assert await database_sync_to_async(activate)(source.doc, source.output,
        text=REVISED, parser="p1-publication-b") == "activated"
    await database_sync_to_async(source.doc.refresh_from_db)()
    current = await database_sync_to_async(lambda: source.doc.chunks.get())()
    assert current.pk != old.pk
    new_citation = f"[doc:{source.doc.id} chunk:{current.pk}]"
    fresh_provider = Provider([cited_reply(REVISED, new_citation)])
    fresh_consumer = consumer_for(source)
    historical = loaded.model_dump()
    fresh_consumer.convo = loaded + [UserMessage(content=QUESTION)]
    async with preservation_turn(fresh_consumer, max_func_calls=3):
        new_message = await retrieve(fresh_consumer, fresh_provider)
        assert new_citation in new_message.content
        assert "α😀 Capacity is 24 litres at 30 C." in new_message.content
        assert "The capacity applies only below 2 Pa." in new_message.content
        assert "18 litres" not in new_message.content and citation not in new_message.content
    assert len(fresh_provider.requests) == 1
    selected_message = fresh_provider.requests[0]["messages"][-1]["content"]
    assert REVISED in selected_message and SOURCE not in selected_message
    assert new_citation in selected_message and citation not in selected_message
    assert loaded.model_dump() == historical
    assert await database_sync_to_async(lambda: source.db.db_messages.get(message_uuid=original_uuid).content)() == saved
    assert await database_sync_to_async(TextChunk.objects.filter(pk=old.pk, content=old.content).exists)()

    def read_history_and_revoke():
        client = Client()
        client.force_login(source.doc.ingested_by)
        url = reverse("api_chunk_detail", kwargs={"chunk_id": old.pk})
        response = client.get(url, {"include_full_text": "1"})
        assert response.status_code == 200
        body = response.json()
        assert body["source_context_status"] == "verified"
        assert body["document"]["full_text"] == SOURCE
        assert (body["start_position"], body["end_position"], body["content"], body["chunk_number"]) == (
            old.start_position, old.end_position, old.content, old.chunk_number)
        assert body["document"]["full_text"][body["start_position"]:body["end_position"]] == body["content"]
        CollectionPermission.objects.filter(user=source.doc.ingested_by, collection=source.doc.collection).delete()
        assert client.get(url).status_code == 403
        assert client.get(reverse("api_chunk_detail", kwargs={"chunk_id": current.pk})).status_code == 403
    await database_sync_to_async(read_history_and_revoke)()
