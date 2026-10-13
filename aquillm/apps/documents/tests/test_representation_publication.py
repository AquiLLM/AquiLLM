"""Real stream/refresh/transcript sinks fence retained but no longer current chunks."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync

from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.refs import CollectionsRef
from apps.chat.services.rag_answer_publication import _freeze_legacy_packet, bind_packet_publication
from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.documents.models import TextChunk
from apps.documents.services.representation_media_acquisition import initialize_media_runtime
from apps.documents.services.source_loading import SourceRuntime, source_query_rows, source_runtime_scope
from aquillm.message_adapters import save_conversation_to_db
from aquillm.models import WSConversation
from lib.llm.answer_publication import AnswerPublicationRejected, bind_answer
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .representation_support import retained as retained, upload as upload, activate

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_generation_binding_at_all_actual_sinks(retained, source_mode, swap):
    from apps.chat.consumers.chat_publish import end_spin_publish
    from lib.llm.providers.final_stream import send_final_answer_stream

    doc, output, old = retained
    budget = TurnBudget(TurnLimits())
    state = TurnContext(budget)
    auth = build_selected_scope_authorization_context(principal=doc.ingested_by,
        selected_collection_ids=(doc.collection_id,), selected_documents=(doc,))
    runtime = SourceRuntime(budget, auth)
    consumer = SimpleNamespace(user=doc.ingested_by, col_ref=CollectionsRef([doc.collection_id]))
    if not source_mode:
        runtime = initialize_media_runtime(consumer, state)
    original_scope = getattr(runtime, "_frozen_retrieval_scope", None)
    citation = f"[doc:{doc.id} chunk:{old.pk}]"
    packet = build_evidence_packet({"result": [{"doc_id": str(doc.id), "chunk_id": old.pk,
        "chunk": 0, "text": old.content, "citation": citation}]}, query="table", search_scope="selected")
    db = WSConversation.objects.create(owner=doc.ingested_by)
    events = []
    async def stream(payload):
        events.append(payload)
    async def send(**kwargs):
        events.append(kwargs)

    with bind_turn(state), source_runtime_scope(runtime) if source_mode else nullcontext():
        if source_mode:
            source_query_rows(TextChunk.objects.filter(pk=old.pk))
        assert _freeze_legacy_packet(consumer, packet,
            (doc.collection_id,), budget)
        if not source_mode:
            assert packet._publication_representation is original_scope.representation
        context = bind_packet_publication(packet)
        text = old.content + " " + citation
        answer = async_to_sync(context.finalize)(text)
        assert answer.outcome != "limited"
        message = bind_answer(AssistantMessage(content=answer.content, stop_reason="stop"), answer, context)
        convo = Conversation(system="s", messages=[message])
        if swap:
            # Same bytes: only the actual generation changed, never the cited row.
            assert activate(doc, output, text=old.content) == "activated"
            assert TextChunk.objects.get(pk=old.pk).content == old.content
        calls = (
            lambda: async_to_sync(send_final_answer_stream)(stream, message_uuid=str(message.message_uuid),
                content=message.content, stop_reason="stop", usage=0, message=message),
            lambda: async_to_sync(end_spin_publish)(SimpleNamespace(convo=convo, _spin_epoch=0, send=send), convo),
            lambda: save_conversation_to_db(convo, db),
        )
        for call in calls:
            if swap:
                with pytest.raises(AnswerPublicationRejected):
                    call()
            else:
                call()
    assert len(events) == (0 if swap else 2)
    assert db.db_messages.count() == (0 if swap else 1)
