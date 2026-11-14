"""Direct synthesis with current PostgreSQL authority and final support behavior."""

from types import SimpleNamespace

import pytest
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.db import connections

from apps.chat.services.rag_answer_publication import bind_legacy_answer_packet
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.chat.services.rag_turn import preservation_turn
from apps.documents.services.source_loading import source_query_rows
from aquillm.message_adapters import save_conversation_to_db
from aquillm.models import (
    Collection,
    CollectionPermission,
    RawTextDocument,
    TextChunk,
    WSConversation,
)
from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.providers import visibility
from lib.llm.turn_context import current_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

pytestmark = pytest.mark.django_db(transaction=True)
SOURCE = "Calibration uses flat fields and dark frames to remove signatures."


class Model:
    def __init__(self, reply):
        self.reply, self.complete_calls = reply, 0

    async def complete(self, convo, max_tokens, stream_func=None):
        self.complete_calls += 1
        return convo + [
            AssistantMessage(content=self.reply, stop_reason="stop")
        ], "changed"


@pytest.fixture
def selected(request):
    source = getattr(request, "param", SOURCE)
    user = get_user_model().objects.create_user(username="support-synthesis")
    collection = Collection.objects.create(name="Synthetic")
    CollectionPermission.objects.create(
        user=user, collection=collection, permission="VIEW"
    )
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
        "text": source,
    }
    return SimpleNamespace(
        consumer=SimpleNamespace(
            user=user, col_ref=SimpleNamespace(collections=[collection.pk])
        ),
        row=row,
        citation=citation,
    )


async def synthesize(
    model, selected, *, bound=True, empty=False, image=False, persist=False
):
    row = dict(selected.row)
    if image:
        row["image_url"] = "/aquillm/document_image/unbound-image/"
    raw = {
        "result": [] if empty else [row],
        "retrieval_status": "no_results" if empty else "results_found",
    }
    packet = build_evidence_packet(
        raw, query="Explain the calibration method.", search_scope="selected documents"
    )
    convo = Conversation(
        system="Use documents",
        messages=[
            UserMessage(content=packet.query),
            ToolMessage(
                content="evidence",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict=raw,
            ),
        ],
    )
    events = []

    async def stream(payload):
        events.append(payload)

    if not bound:
        return await synthesize_from_evidence(model, convo, packet), events
    selected.consumer.convo = convo
    try:
        async with preservation_turn(selected.consumer, max_func_calls=3):
            state = current_turn()
            runtime = state._media_source_runtime
            scope = runtime._frozen_retrieval_scope

            def acquire():
                rows = source_query_rows(
                    TextChunk.objects.filter(pk=selected.row["chunk_id"]),
                    runtime=runtime, require_existing=True,
                )
                assert len(rows) == 1 and rows[0].content == selected.row["text"]
                assert runtime._frozen_retrieval_scope is scope
                assert runtime.budget.text_used["materialized"] >= len(selected.row["text"])

            await database_sync_to_async(acquire)()
            await bind_legacy_answer_packet(selected.consumer, packet)
            result = await synthesize_from_evidence(
                model, convo, packet, stream_func=stream, max_tokens=1024
            )
            answer = result[-1]._validated_answer
            await state.answer_publication.verify(
                answer, content=result[-1].content, sink="persist"
            )
            if persist:

                def save_answer():
                    db = WSConversation.objects.create(owner=selected.consumer.user)
                    save_conversation_to_db(result, db)
                    return db.db_messages.get(
                        message_uuid=result[-1].message_uuid
                    ).content

                saved = await database_sync_to_async(save_answer)()
                assert saved == result[-1].content
            assert runtime._frozen_retrieval_scope is scope
    finally:
        await database_sync_to_async(connections.close_all)()
    assert len(events) == 1
    assert events[0]["content"] == result[-1].content
    return result, events


async def test_cited_model_paraphrase_reaches_stream_and_database_in_one_call(selected):
    # The provider supplies the wording; citation validity does not promote its
    # semantic support state or replace the answer with a deterministic excerpt.
    draft = (
        "The calibration method removes signatures using flat fields and dark "
        f"frames {selected.citation}."
    )
    model = Model(draft)
    result, events = await synthesize(model, selected, persist=True)
    assert model.complete_calls == 1
    assert result[-1].content == draft
    assert events[0]["content"] == draft
    assert result[-1]._validated_answer.outcome == "partial"


async def test_blank_synthesis_uses_extractive_fallback(selected, monkeypatch):
    monkeypatch.delenv("LLM_ALLOW_EXTRACTIVE_EVIDENCE_UI", raising=False)
    result, _ = await synthesize(Model(""), selected)
    assert f"> {SOURCE} {selected.citation}" in result[-1].content
    assert "uncertain" in result[-1].content


async def test_failure_text_replaced_by_extractive(selected):
    failure = visibility.clean_response_failure_text(after_tool_result=True)
    result, _ = await synthesize(Model(failure), selected)
    assert result[-1].content != failure
    assert f"> {SOURCE} {selected.citation}" in result[-1].content


async def test_no_results_returns_scoped_notice_without_llm(selected):
    model = Model("must not run")
    result, _ = await synthesize(model, selected, empty=True)
    assert model.complete_calls == 0
    assert result[-1].content == LIMITED_ANSWER
    assert "does not establish" in result[-1].content


@pytest.mark.parametrize(
    "draft",
    [
        "Calibration removes signatures.",
        "Calibration removes signatures [doc:foreign chunk:9].",
    ],
)
async def test_missing_or_invalid_citations_use_complete_source_fallback(selected, draft):
    model = Model(draft)
    result, _ = await synthesize(model, selected)
    assert model.complete_calls == 1
    assert f"> {SOURCE} {selected.citation}" in result[-1].content
    assert "doc:foreign" not in result[-1].content
    assert result[-1]._validated_answer.outcome == "partial"


async def test_no_figure_appended_when_not_requested(selected):
    draft = f"Calibration uses flat fields {selected.citation}."
    result, _ = await synthesize(Model(draft), selected)
    assert "![" not in result[-1].content
    assert result[-1].content == draft


async def test_model_selected_image_without_immutable_authority_is_limited(selected):
    reply = (
        f"Selected figure {selected.citation}.\n"
        "![image](/aquillm/document_image/unbound-image/)"
    )
    result, _ = await synthesize(Model(reply), selected)
    assert result[-1].content == LIMITED_ANSWER
    assert "![" not in result[-1].content


async def test_retrieved_image_without_current_authority_is_limited(selected):
    result, _ = await synthesize(
        Model(f"Calibration {selected.citation}."), selected, image=True
    )
    assert result[-1].content == LIMITED_ANSWER


async def test_extractive_fallback_does_not_select_a_figure(selected):
    result, _ = await synthesize(Model(""), selected)
    assert f"> {SOURCE} {selected.citation}" in result[-1].content
    assert "![" not in result[-1].content


async def test_unbound_synthesis_cannot_publish_selected_text(selected):
    result, events = await synthesize(
        Model(f"Calibration {selected.citation}."), selected, bound=False
    )
    assert result[-1].content == LIMITED_ANSWER
    assert events == []
