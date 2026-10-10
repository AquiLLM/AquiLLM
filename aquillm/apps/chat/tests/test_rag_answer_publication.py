"""Publication through real stream, refresh, and PostgreSQL transcript sinks."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model

from apps.chat.services.rag_evidence import EvidencePacket
from aquillm.message_adapters import load_conversation_from_db, save_conversation_to_db
from aquillm.models import WSConversation
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage
from lib.retrieval.synthesis_budget import seal_synthesis
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


def setup_publication(*, image_urls=()):
    from apps.chat.services.rag_answer_publication import RagAnswerPublication

    clock = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: clock[0])
    state = TurnContext(budget)
    packet = EvidencePacket(
        [{"doc_id": "a", "chunk_id": 1, "text": "Capacity is 18 litres."}],
        list(image_urls),
        ["[doc:a chunk:1]"],
        "capacity",
        "selected",
        "results_found",
        "",
        10,
    )
    allowed = [True]

    async def authorize():
        return allowed[0]

    lease = seal_synthesis(
        budget, "frozen", calls=3, output_tokens=4096, timeout_seconds=60
    )
    context = RagAnswerPublication(
        packet=packet, authorize=authorize, turn=state, lease=lease
    )
    state.answer_publication = context
    return state, context, packet, allowed, clock


def test_selected_image_without_immutable_bytes_is_limited_at_all_sinks():
    from apps.chat.consumers.chat_publish import end_spin_publish
    from lib.llm.answer_publication import LIMITED_ANSWER
    from lib.llm.providers.final_stream import send_final_answer_stream

    url = "/aquillm/document_image/11111111-1111-4111-8111-111111111111/"
    state, context, _, _, _ = setup_publication(image_urls=(url,))
    db = WSConversation.objects.create(
        owner=get_user_model().objects.create_user(username="publication")
    )
    events = []

    async def stream(payload):
        events.append(payload["content"])

    async def send(**kwargs):
        events.append(
            json.loads(kwargs["text_data"])["delta"]["messages"][0]["content"]
        )

    with bind_turn(state):
        message = message_for(
            context,
            f"Capacity is 18 litres [doc:a chunk:1].\n![Selected figure]({url})",
        )
        convo = Conversation(system="s", messages=[message])
        async_to_sync(send_final_answer_stream)(
            stream,
            message_uuid=str(message.message_uuid),
            content=message.content,
            stop_reason="stop",
            usage=0,
            message=message,
        )
        async_to_sync(end_spin_publish)(
            SimpleNamespace(convo=convo, _spin_epoch=0, send=send), convo
        )
        save_conversation_to_db(convo, db)
    assert events == [LIMITED_ANSWER, LIMITED_ANSWER]
    assert db.db_messages.get().content == LIMITED_ANSWER


def message_for(context, text="Capacity is 18 litres [doc:a chunk:1]."):
    from lib.llm.answer_publication import bind_answer

    return bind_answer(
        AssistantMessage(content=text, stop_reason="stop"),
        async_to_sync(context.finalize)(text),
        context,
    )


@pytest.mark.parametrize(
    "change", ["revoked", "revision", "scope", "content", "closed", "receipt"]
)
def test_each_actual_sink_rechecks_frozen_binding(change):
    from apps.chat.consumers.chat_publish import end_spin_publish
    from lib.llm.answer_publication import AnswerPublicationRejected
    from lib.llm.providers.final_stream import send_final_answer_stream

    state, context, packet, allowed, _ = setup_publication()
    db = WSConversation.objects.create(
        owner=get_user_model().objects.create_user(username="publication")
    )
    events = []

    async def send(**kwargs):
        events.append(kwargs)

    async def stream(payload):
        events.append(payload)

    with bind_turn(state):
        message = message_for(context)
        convo = Conversation(system="s", messages=[message])
        if change == "revoked":
            allowed[0] = False
        elif change == "revision":
            packet.chunks[0]["text"] = "changed source"
        elif change == "scope":
            packet.search_scope = "other documents"
        elif change == "content":
            message.content += " Invented extra."
        elif change == "closed":
            state.budget.close("cancelled")
        elif change == "receipt":
            message._validated_answer = None
        with pytest.raises((AnswerPublicationRejected, asyncio.CancelledError)):
            async_to_sync(send_final_answer_stream)(
                stream,
                message_uuid=str(message.message_uuid),
                content=message.content,
                stop_reason="stop",
                usage=0,
                message=message,
            )
        with pytest.raises((AnswerPublicationRejected, asyncio.CancelledError)):
            async_to_sync(end_spin_publish)(
                SimpleNamespace(convo=convo, _spin_epoch=0, send=send), convo
            )
        with pytest.raises((AnswerPublicationRejected, asyncio.CancelledError)):
            save_conversation_to_db(convo, db)
    assert events == []
    assert db.db_messages.count() == 0


def test_exact_content_agrees_after_retrieval_deadline_and_history_has_no_receipt():
    from apps.chat.consumers.chat_publish import end_spin_publish
    from lib.llm.providers.final_stream import send_final_answer_stream

    state, context, _, _, clock = setup_publication()
    db = WSConversation.objects.create(
        owner=get_user_model().objects.create_user(username="publication")
    )
    events = []

    async def stream(payload):
        events.append(payload)

    async def send(**kwargs):
        events.append(json.loads(kwargs["text_data"])["delta"]["messages"][0])

    with bind_turn(state):
        message = message_for(context)
        clock[0] = 20.0
        convo = Conversation(system="s", messages=[message])
        async_to_sync(send_final_answer_stream)(
            stream,
            message_uuid=str(message.message_uuid),
            content=message.content,
            stop_reason="stop",
            usage=0,
            message=message,
        )
        async_to_sync(end_spin_publish)(
            SimpleNamespace(convo=convo, _spin_epoch=0, send=send), convo
        )
        save_conversation_to_db(convo, db)
        assert all(event["content"] == message.content for event in events)
        assert db.db_messages.get().content == message.content
        assert "receipt" not in message.model_dump_json()
        loaded = load_conversation_from_db(db)
        assert loaded[-1]._validated_answer is None
    assert context._receipts == {}


def test_revocation_during_finalize_yields_only_fact_free_template():
    from lib.llm.answer_publication import LIMITED_ANSWER

    state, context, _, allowed, _ = setup_publication()
    with bind_turn(state):
        allowed[0] = False
        message = message_for(context)
        assert message.content == LIMITED_ANSWER
        assert "doc:" not in message.content
        assert message._validated_answer.outcome == "limited"


def test_receipt_cannot_be_reused_in_another_turn():
    state, context, _, _, _ = setup_publication()
    with bind_turn(state):
        answer = async_to_sync(context.finalize)(
            "Capacity is 18 litres [doc:a chunk:1]."
        )
        other_state, other, _, _, _ = setup_publication()
        with bind_turn(other_state), pytest.raises(Exception, match="unbound_receipt"):
            async_to_sync(other.verify)(answer, content=answer.content, sink="stream")


@pytest.mark.parametrize(
    "failure", ["save", "revoke_after_save", "reload_receipt", "reload_persist"]
)
def test_delta_or_snapshot_never_bypasses_failed_save_or_lost_receipt(
    failure, monkeypatch
):
    from channels.db import database_sync_to_async

    from apps.chat.consumers.chat_delta import (
        send_conversation_delta,
        send_conversation_snapshot,
    )
    from lib.llm.answer_publication import AnswerPublicationRejected

    state, context, _, allowed, _ = setup_publication()
    db = WSConversation.objects.create(
        owner=get_user_model().objects.create_user(username="publication")
    )
    events = []

    async def send(**kwargs):
        events.append(kwargs)

    with bind_turn(state):
        convo = Conversation(system="s", messages=[message_for(context)])
        consumer = SimpleNamespace(
            convo=convo,
            db_convo=db,
            last_sent_sequence=-1,
            send=send,
            transport_connected=True,
        )

        async def save(**kwargs):
            await database_sync_to_async(save_conversation_to_db)(consumer.convo, db)
            if failure == "revoke_after_save":
                allowed[0] = False

        consumer._save_conversation = save
        if failure == "save":
            from aquillm.models import Message

            def broken_write(*args, **kwargs):
                raise RuntimeError("synthetic write failure")

            monkeypatch.setattr(Message.objects, "bulk_create", broken_write)
        if failure in {"reload_receipt", "reload_persist"}:
            save_conversation_to_db(convo, db)
            consumer.convo = load_conversation_from_db(db)
            with pytest.raises(AnswerPublicationRejected, match="unbound_receipt"):
                if failure == "reload_receipt":
                    async_to_sync(send_conversation_snapshot)(consumer)
                else:
                    save_conversation_to_db(consumer.convo, db)
        else:
            with pytest.raises((AnswerPublicationRejected, RuntimeError)):
                async_to_sync(send_conversation_delta)(consumer, convo)
    assert events == []
    if failure == "save":
        assert db.db_messages.count() == 0
