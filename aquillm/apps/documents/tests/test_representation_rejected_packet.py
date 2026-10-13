"""Invalid default evidence is stopped before provider and every output sink."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth.models import User

from apps.chat.refs import CollectionsRef
from apps.chat.services.rag_answer_publication import bind_legacy_answer_packet
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.collections.models import CollectionPermission
from apps.documents.services.representation_media_acquisition import initialize_media_runtime
from aquillm.message_adapters import save_conversation_to_db
from aquillm.models import WSConversation
from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage, UserMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .representation_support import retained as retained, upload as upload

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("invalid", [False, True])
def test_empty_or_invalid_default_packet_has_no_provider_or_source_egress(retained, invalid):
    from apps.chat.consumers.chat_publish import end_spin_publish
    from lib.llm.providers.final_stream import send_final_answer_stream
    doc, _, chunk = retained
    user = User.objects.create_user(username="rejected-packet-viewer")
    CollectionPermission.objects.create(user=user, collection=doc.collection, permission="VIEW")
    sentinel = "SUBSTITUTED_UNVERIFIED_SOURCE"
    raw = {"result": [{"doc_id": str(doc.id), "chunk_id": chunk.pk,
        "chunk": chunk.chunk_number, "citation": f"[doc:{doc.id} chunk:{chunk.pk}]",
        "text": sentinel}] if invalid else []}
    convo = Conversation(system="s", messages=[UserMessage(content="Explain source"),
        ToolMessage(tool_name="vector_search", for_whom="assistant", content=str(raw), result_dict=raw)])
    consumer = SimpleNamespace(user=user, col_ref=CollectionsRef([doc.collection_id]), convo=convo)
    turn = TurnContext(TurnBudget(TurnLimits()))
    initialize_media_runtime(consumer, turn)
    packet = build_evidence_packet(raw, query="source", search_scope="selected")
    provider = SimpleNamespace(complete=AsyncMock())
    events = []
    async def stream(payload):
        events.append(payload)
    async def send(**kwargs):
        events.append(kwargs)
    transcript = WSConversation.objects.create(owner=user)
    with bind_turn(turn):
        async_to_sync(bind_legacy_answer_packet)(consumer, packet)
        assert packet.source_authorization is None and packet.retrieval_status == "context_limited"
        answer = async_to_sync(synthesize_from_evidence)(provider, convo, packet)
        provider.complete.assert_not_called()
        assert answer[-1].content == LIMITED_ANSWER
        assert sentinel not in str(answer.model_dump())
        message = answer[-1]
        async_to_sync(send_final_answer_stream)(stream, message_uuid=str(message.message_uuid),
            content=message.content, stop_reason="stop", usage=0, message=message)
        async_to_sync(end_spin_publish)(SimpleNamespace(convo=answer, _spin_epoch=0, send=send), answer)
        save_conversation_to_db(answer, transcript)
    assert len(events) == 2 and sentinel not in str(events)
    assert sentinel not in str(list(transcript.db_messages.values()))
    assert not getattr(turn, "_manual_publication_bindings", {})
