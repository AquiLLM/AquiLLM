"""A live turn cannot invent acquisition authority at final publication."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth.models import User

from apps.chat.refs import CollectionsRef
from apps.chat.services.rag_answer_publication import (
    _freeze_legacy_packet, bind_legacy_answer_packet, bind_packet_publication,
)
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.collections.models import CollectionPermission
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_media_acquisition import initialize_media_runtime
from aquillm.message_adapters import save_conversation_to_db
from aquillm.models import WSConversation
from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.providers.openai import OpenAIInterface
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage, UserMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .representation_support import retained as retained, upload as upload

pytestmark = pytest.mark.django_db(transaction=True)


def _source(retained):
    doc, _, chunk = retained
    user = User.objects.create_user(username="missing-capture-viewer")
    CollectionPermission.objects.create(user=user, collection=doc.collection, permission="VIEW")
    citation = f"[doc:{doc.id} chunk:{chunk.pk}]"
    raw = {"result": [{"doc_id": str(doc.id), "chunk_id": chunk.pk,
        "chunk": chunk.chunk_number, "citation": citation, "text": chunk.content}]}
    convo = Conversation(system="s", messages=[UserMessage(content="Explain source"),
        ToolMessage(tool_name="vector_search", for_whom="assistant", content=str(raw), result_dict=raw)])
    consumer = SimpleNamespace(user=user, col_ref=CollectionsRef([doc.collection_id]), convo=convo)
    return doc, chunk, consumer, build_evidence_packet(raw, query="source", search_scope="selected")


@pytest.mark.parametrize("remove_original", [False, True])
def test_missing_carrier_stops_actual_sdk_and_every_output_sink(retained, remove_original):
    from apps.chat.consumers.chat_publish import end_spin_publish
    from lib.llm.providers.final_stream import send_final_answer_stream

    doc, chunk, consumer, packet = _source(retained)
    turn = TurnContext(TurnBudget(TurnLimits()))
    original = initialize_media_runtime(consumer, turn) if remove_original else None
    snapshot = original._frozen_retrieval_scope.representation if original else None
    if original:
        assert snapshot.selections and turn._media_source_runtime is original
        del turn._media_source_runtime
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=chunk.content + f" [doc:{doc.id} chunk:{chunk.pk}]", tool_calls=None),
        finish_reason="stop")], usage=None))
    provider = OpenAIInterface(SimpleNamespace(base_url="http://fake-models:8000/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))), "test")
    events = []
    async def stream(payload):
        events.append(payload)
    async def send(**kwargs):
        events.append(kwargs)
    transcript = WSConversation.objects.create(owner=consumer.user)
    with bind_turn(turn):
        async_to_sync(bind_legacy_answer_packet)(consumer, packet)
        answer = async_to_sync(synthesize_from_evidence)(provider, consumer.convo, packet)
        create.assert_not_called()
        assert answer[-1].content == LIMITED_ANSWER
        assert packet.source_authorization is None and packet.retrieval_status == "context_limited"
        assert chunk.content not in str(answer.model_dump())
        message = answer[-1]
        async_to_sync(send_final_answer_stream)(stream, message_uuid=str(message.message_uuid),
            content=message.content, stop_reason="stop", usage=0, message=message)
        async_to_sync(end_spin_publish)(SimpleNamespace(convo=answer, _spin_epoch=0, send=send), answer)
        save_conversation_to_db(answer, transcript)
        assert not hasattr(turn, "_media_source_runtime")
    assert len(events) == 2 and chunk.content not in str(events)
    assert chunk.content not in str(list(transcript.db_messages.values()))
    if original:
        assert original._frozen_retrieval_scope.representation is snapshot


def test_outside_turn_helper_cannot_mint_nonnull_publication_in_bare_live_turn(retained):
    doc, _, consumer, packet = _source(retained)
    turn = TurnContext(TurnBudget(TurnLimits()))
    # Genuine no-turn helper compatibility does not authorize a subsequent turn.
    assert _freeze_legacy_packet(consumer, packet, (doc.collection_id,), turn.budget)
    assert packet.source_authorization is not None
    with bind_turn(turn):
        with pytest.raises(SourceScopeUnavailable, match="source_scope_unavailable"):
            bind_packet_publication(packet)
        assert turn.answer_publication is None
