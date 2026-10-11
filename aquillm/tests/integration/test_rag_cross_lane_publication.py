"""Original real retrieval authority is rechecked at every actual answer sink."""
from types import SimpleNamespace

import pytest
from channels.db import database_sync_to_async

from apps.chat.consumers.chat_publish import end_spin_publish
from apps.chat.services.rag_turn import preservation_turn
from apps.collections.models import CollectionPermission
from apps.documents.services.representation_media_acquisition import media_runtime
from apps.documents.tests.representation_support import (
    activate, retained as retained, upload as upload,
)
from aquillm.message_adapters import save_conversation_to_db
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.answer_publication import AnswerPublicationRejected
from lib.llm.providers.final_stream import send_final_answer_stream
from lib.llm.turn_context import current_turn
from .cross_lane_publication_support import (
    Provider, REVISED, SOURCE, cited_reply, consumer_for, publish, retrieve,
    close_publication_worker as close_publication_worker,
    publication_source as publication_source,
)

pytestmark = pytest.mark.django_db(transaction=True)


def change_source(source, change):
    if change == "revoke":
        CollectionPermission.objects.filter(user=source.user, collection=source.doc.collection).delete()
    elif change == "edit":
        source.doc.full_text = REVISED
        source.doc.save(dont_rechunk=True)
    elif change == "generation":
        assert activate(source.doc, source.output, text=SOURCE, parser="p1-same-text-generation") == "activated"
        source.doc.refresh_from_db()
        assert source.doc.full_text == SOURCE
        assert source.doc.chunks.get().pk != source.chunk.pk
    else:
        raise AssertionError("unknown transition")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "source"])
@pytest.mark.parametrize("change", [None, "revoke", "edit", "generation"])
async def test_actual_retrieval_receipt_fences_every_sink(publication_source, monkeypatch, mode, change):
    source = publication_source
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", mode)
    citation = f"[doc:{source.doc.id} chunk:{source.chunk.pk}]"
    provider = Provider([cited_reply(SOURCE, citation)])
    consumer = consumer_for(source)
    events = []
    async def stream(payload):
        events.append(payload)
    async def send(**kwargs):
        events.append(kwargs)
    async with preservation_turn(consumer, max_func_calls=3):
        turn, runtime = current_turn(), media_runtime()
        frozen = runtime._frozen_retrieval_scope.representation
        message = await retrieve(consumer, provider)
        context, answer = turn.answer_publication, message._validated_answer
        assert answer.outcome == "partial" and citation in message.content
        assert "18 litres at 20 C" in message.content and "only below 1 Pa" in message.content
        assert len(provider.requests) == 1 and not provider.answers
        if change is None:
            await publish(source, consumer)
            return
        await database_sync_to_async(change_source)(source, change)
        context._active()
        assert context.remaining_seconds() >= 2
        with pytest.raises(AnswerPublicationRejected, match="^authority_unavailable$"):
            await send_final_answer_stream(stream, message_uuid=str(message.message_uuid),
                content=message.content, stop_reason=message.stop_reason, usage=message.usage, message=message)
        context._active()
        assert context.remaining_seconds() >= 2
        with pytest.raises(AnswerPublicationRejected, match="^authority_unavailable$"):
            await end_spin_publish(SimpleNamespace(convo=consumer.convo,
                _spin_epoch=len(consumer.convo)-1, send=send), consumer.convo)
        context._active()
        assert context.remaining_seconds() >= 2
        with pytest.raises(AnswerPublicationRejected, match="^authority_unavailable$"):
            await database_sync_to_async(save_conversation_to_db)(consumer.convo, source.db)
        assert events == [] and await database_sync_to_async(source.db.db_messages.count)() == 0
        assert current_turn() is turn and turn.answer_publication is context
        assert turn._media_source_runtime is runtime
        assert runtime._frozen_retrieval_scope.representation is frozen
        assert message._validated_answer is answer
