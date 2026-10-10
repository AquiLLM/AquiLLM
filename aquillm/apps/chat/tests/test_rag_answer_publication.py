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


@pytest.mark.parametrize("owner", ["budget", "execution"])
def test_cancellation_during_actual_write_rolls_back_transaction(owner, monkeypatch):
    from uuid import uuid4

    from apps.chat.services.execution import Execution
    from aquillm.models import Message
    from lib.llm.execution_context import bind_execution

    db = WSConversation.objects.create(
        owner=get_user_model().objects.create_user(username="publication")
    )
    execution = Execution(db.pk, uuid4())
    with bind_execution(execution):
        state, context, _, _, _ = setup_publication()
        with bind_turn(state):
            convo = Conversation(system="s", messages=[message_for(context)])
            original = Message.objects.bulk_create

            def close_after_write(*args, **kwargs):
                result = original(*args, **kwargs)
                if owner == "budget":
                    state.budget.close("cancelled")
                else:
                    execution.cancel()
                return result

            monkeypatch.setattr(Message.objects, "bulk_create", close_after_write)
            with pytest.raises(asyncio.CancelledError):
                save_conversation_to_db(convo, db)
    assert db.db_messages.count() == 0


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


@pytest.mark.parametrize(
    "change", [None, "revoke", "revision", "scope", "forged", "db_timeout"]
)
@pytest.mark.parametrize(
    "route", ["direct", "whole_document", "figure_caption", "figure_chunks"]
)
def test_legacy_direct_synthesis_binds_actual_selected_database_source(
    change, route, monkeypatch
):
    from apps.chat.services.rag_answer_publication import bind_legacy_answer_packet
    from apps.chat.services.rag_synthesis import synthesize_from_evidence
    from aquillm.models import (
        Collection,
        CollectionPermission,
        RawTextDocument,
        TextChunk,
    )
    from lib.llm.answer_publication import LIMITED_ANSWER
    from lib.llm.providers.complete_turn import complete_conversation_turn
    from lib.llm.types.messages import ToolMessage, UserMessage
    from lib.llm.types.response import LLMResponse

    user = get_user_model().objects.create_user(username="legacy-publication")
    collection = Collection.objects.create(name="Synthetic")
    permission = CollectionPermission.objects.create(
        user=user, collection=collection, permission="VIEW"
    )
    source = "Capacity is 18 litres."
    doc = RawTextDocument.objects.create(
        title="Synthetic source",
        full_text=source,
        collection=collection,
        ingested_by=user,
    )
    chunk = TextChunk.objects.create(
        content=source,
        start_position=0,
        end_position=len(source),
        chunk_number=0,
        doc_id=doc.id,
    )
    citation = f"[doc:{doc.id} chunk:{chunk.pk}]"
    row = {
        "doc_id": str(doc.id),
        "chunk_id": chunk.pk,
        "chunk": 0,
        "citation": citation,
        "text": source if change != "forged" else "Forged excerpt.",
    }
    packet = EvidencePacket(
        [row], [], [citation], "capacity", "selected documents", "results_found", "", 10
    )
    figure = figure_chunk = None
    if route.startswith("figure_"):
        from apps.documents.models import DocumentFigure
        from lib.llm.answer_publication import content_digest

        figure = DocumentFigure(
            title="Synthetic caption",
            collection=collection,
            ingested_by=user,
            image_file="synthetic/unread.png",
            extracted_caption=source,
            full_text=source,
            figure_index=0,
        )
        figure.parent_document = doc
        figure.save()
        chunk_ids = []
        if route == "figure_chunks":
            figure_chunk = TextChunk.objects.create(
                content=source,
                start_position=0,
                end_position=len(source),
                chunk_number=0,
                doc_id=figure.id,
            )
            chunk_ids = [figure_chunk.pk]
        packet.image_urls = [f"/aquillm/document_image/{figure.id}/"]
        packet.auxiliary_figures = (
            {"image_url": packet.image_urls[0], "text": source},
        )
        packet.auxiliary_handoff = (
            {
                "_figure_provenance": [
                    {
                        "figure_id": str(figure.id),
                        "parent_id": str(doc.id),
                        "field": "extracted_caption",
                        "source_fingerprint": content_digest(source),
                        "chunk_ids": chunk_ids,
                    }
                ]
            },
            user,
        )
    state = TurnContext(TurnBudget(TurnLimits()))
    consumer = SimpleNamespace(
        user=user, col_ref=SimpleNamespace(collections=[collection.pk])
    )
    convo = Conversation(
        system="Use evidence",
        messages=[
            UserMessage(content="Capacity?"),
            ToolMessage(
                content=source,
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={"result": [row]},
            ),
        ],
    )
    events = []

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            from channels.db import database_sync_to_async

            if change == "revoke":
                await database_sync_to_async(permission.delete)()
            if change == "revision":
                if figure is not None and figure_chunk is None:
                    await database_sync_to_async(
                        lambda: DocumentFigure.objects.filter(pk=figure.pk).update(
                            extracted_caption="Changed caption."
                        )
                    )()
                else:
                    await database_sync_to_async(
                        lambda: TextChunk.objects.filter(
                            pk=(figure_chunk or chunk).pk
                        ).update(content="Changed source.")
                    )()
            if change == "scope":
                consumer.col_ref.collections = []
            if change == "db_timeout":
                from itertools import count

                from apps.chat.services import rag_answer_publication

                ticks = count(step=2)
                monkeypatch.setattr(
                    rag_answer_publication, "monotonic", lambda: next(ticks)
                )
            return LLMResponse(
                text=f"Capacity is 18 litres {citation}.",
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
            )

        async def complete(self, request, budget, stream_func=None):
            return await complete_conversation_turn(
                self, request, budget, stream_func=stream_func
            )

    async def stream(payload):
        events.append(payload)

    with bind_turn(state):
        if change is None and route == "direct":
            from apps.chat.services.rag_answer_publication import _freeze_legacy_packet

            assert _freeze_legacy_packet(
                consumer, packet, tuple(consumer.col_ref.collections), state.budget
            )
        if route == "whole_document":
            from apps.chat.services.rag_answer_publication import legacy_tool_handoff
            from apps.chat.services.tool_wiring.document_tool_support import (
                format_whole_document_citations,
            )

            text, citations = format_whole_document_citations(doc.id, [chunk])
            if change == "forged":
                text = text.replace(source, "Forged excerpt.")
            convo[-1].tool_name = "whole_document"
            convo[-1].arguments = {"doc_id": str(doc.id)}
            convo[-1].result_dict = {"result": text, "citation_chunks": citations}
            result, _ = async_to_sync(legacy_tool_handoff)(
                consumer, Model(), convo, 1024, stream
            )
        else:
            async_to_sync(bind_legacy_answer_packet)(consumer, packet)
            result = async_to_sync(synthesize_from_evidence)(
                Model(), convo, packet, stream_func=stream, max_tokens=1024
            )
        assert result[-1].content == (
            f"Capacity is 18 litres {citation}." if change is None else LIMITED_ANSWER
        )
        db = WSConversation.objects.create(owner=user)
        save_conversation_to_db(result, db)
        assert (
            db.db_messages.order_by("sequence_number").last().content
            == result[-1].content
        )
        assert events and all(
            event["content"] == result[-1].content for event in events
        )


def test_legacy_historical_tool_is_not_rebound(monkeypatch):
    from apps.chat.services.rag_turn import preservation_turn
    from lib.llm.turn_context import current_turn
    from lib.llm.types.messages import ToolMessage

    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "legacy")
    monkeypatch.setenv("RAG_RERANK_TEXT_MODE", "legacy")
    monkeypatch.setenv("RAG_DOCUMENT_CAPACITY_MODE", "legacy")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "0")
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "0")
    convo = Conversation(
        system="s",
        messages=[
            ToolMessage(
                tool_name="whole_document",
                content="Historical evidence",
                result_dict={"result": "Historical evidence"},
                for_whom="assistant",
            )
        ],
    )
    consumer = SimpleNamespace(convo=convo)

    async def check():
        async with preservation_turn(consumer, max_func_calls=3):
            assert await current_turn().handoff(None, convo, 1024, None) is None
            assert current_turn().answer_publication is None

    async_to_sync(check)()


def test_existing_publication_never_authorizes_later_foreign_tool(monkeypatch):
    from apps.chat.services.rag_answer_publication import RagAnswerPublication
    from apps.chat.services.rag_turn import preservation_turn
    from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
    from lib.llm.answer_publication import AnswerPublicationRejected
    from lib.llm.turn_context import current_turn
    from lib.llm.types.messages import ToolMessage

    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "legacy")
    consumer = SimpleNamespace(convo=Conversation(system="s", messages=[]))
    foreign = consumer.convo + [
        ToolMessage(
            tool_name="whole_document",
            content="Foreign source",
            for_whom="assistant",
            result_dict={"result": "Foreign source"},
        )
    ]
    llm = _FakeLLMInterface([])
    events = []

    async def stream(payload):
        events.append(payload)

    async def check():
        async with preservation_turn(consumer, max_func_calls=3):
            state = current_turn()
            packet = EvidencePacket([], [], [], "", "", "no_results", "", 0)

            async def authorize():
                return True

            context = RagAnswerPublication(
                packet=packet, authorize=authorize, turn=state
            )
            state.answer_publication = context
            with pytest.raises(AnswerPublicationRejected, match="evidence_changed"):
                await llm.complete(foreign, 1024, stream_func=stream)
            assert state.answer_publication is context
            assert not context.issued_messages

    async_to_sync(check)()
    assert llm.calls == []
    assert events == []
