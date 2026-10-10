"""Real default-mode source authority and current tool handoff publication."""

from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model

from apps.chat.services.rag_evidence import EvidencePacket
from aquillm.message_adapters import save_conversation_to_db
from aquillm.models import WSConversation
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


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

                from apps.chat.services import rag_publication_database

                ticks = count(step=2)
                monkeypatch.setattr(
                    rag_publication_database, "monotonic", lambda: next(ticks)
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
        if change is None:
            assert "uncertain" in result[-1].content
            assert f"> Capacity is 18 litres. {citation}" in result[-1].content
            assert result[-1]._validated_answer.outcome == "partial"
        else:
            assert result[-1].content == LIMITED_ANSWER
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
