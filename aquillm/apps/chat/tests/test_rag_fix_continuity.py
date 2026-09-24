from types import SimpleNamespace

import pytest

from apps.chat.refs import CollectionsRef
from apps.chat.tests.test_rag_preservation_routes import configure
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage

pytestmark = [pytest.mark.asyncio, pytest.mark.django_db(transaction=True)]


@pytest.mark.parametrize("route", ["direct", "normal"])
@pytest.mark.parametrize("prior_available", [True, False])
async def test_partial_followup_keeps_prior_and_recoverable_current_evidence(
    docs, monkeypatch, route, prior_available
):
    from apps.chat.services.rag_pipeline import run_direct_rag_turn
    from apps.chat.services.rag_turn import preservation_turn
    from apps.documents.models import TextChunk
    from lib.llm.types.messages import ToolMessage

    configure(monkeypatch)
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "1")
    user, doc, chunks, _ = docs
    missing = "11111111-1111-4111-8111-111111111111"
    rows = [
        {
            "doc_id": str(doc.id),
            "chunk_id": chunks[0].pk,
            "chunk": 0,
            "title": "Available",
            "text": "historical",
        },
        {
            "doc_id": missing,
            "chunk_id": 9999999,
            "chunk": 0,
            "title": "Missing",
            "text": "missing old text",
        },
    ]
    for row in rows:
        row["citation"] = f"[doc:{row['doc_id']} chunk:{row['chunk_id']}]"
    if not prior_available:
        rows = rows[1:]
    question = (
        "Compare both documents and report the current measurement"
        if prior_available
        else f"Explain {rows[0]['citation']} and report the current measurement"
    )
    convo = Conversation(
        system="sys",
        messages=[
            ToolMessage(
                tool_name="vector_search",
                for_whom="assistant",
                content="historical",
                arguments={},
                result_dict={"result": rows},
            ),
            AssistantMessage(
                content=" ".join(r["citation"] for r in rows), stop_reason="end_turn"
            ),
            UserMessage(content=question),
        ],
    )
    from apps.documents.services.source_loading import source_query_rows

    def search(*args, **kwargs):
        return (
            None,
            None,
            source_query_rows(TextChunk.objects.filter(pk=chunks[1].pk)),
            {},
        )

    monkeypatch.setattr(TextChunk, "text_chunk_search", search)
    requests = []

    async def complete(request, *args, **kwargs):
        requests.append(request)
        return request + [
            AssistantMessage(
                content="One available source; the other is unavailable.",
                stop_reason="end_turn",
            )
        ], "changed"

    consumer = SimpleNamespace(
        user=user, col_ref=CollectionsRef([doc.collection_id]), convo=convo
    )
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        if route == "direct":
            await run_direct_rag_turn(
                consumer, SimpleNamespace(complete=complete), convo
            )
        else:
            from channels.db import database_sync_to_async

            from apps.chat.services.rag_preservation_turn import normal_source_handoff
            from apps.chat.services.tool_wiring.documents import vector_search_tool

            raw = await database_sync_to_async(
                vector_search_tool(user, consumer.col_ref)
            )(search_string="current measurement", top_k=5)
            working = convo + [
                ToolMessage(
                    tool_name="vector_search",
                    for_whom="assistant",
                    content="empty",
                    arguments={},
                    result_dict=raw,
                )
            ]
            await normal_source_handoff(
                consumer, SimpleNamespace(complete=complete), working, 2048, None
            )
        assert runtime.budget.actions_used == 1
        assert runtime.budget.sources_used == (2 if prior_available else 1)
    assert len(requests) == 1
    assert (chunks[0].content in requests[0][-1].content) is prior_available
    assert chunks[1].content in requests[0][-1].content
    assert "missing old text" not in requests[0][-1].content
    assert "no longer access" in requests[0][-1].content
    assert "Do not substitute" in requests[0][-1].content
