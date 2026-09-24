"""Normal whole/adjacent handoff revalidates current chunk and figure authority."""

from types import SimpleNamespace

import pytest
from channels.db import database_sync_to_async

from apps.chat.refs import CollectionsRef
from apps.chat.tests.test_rag_preservation_routes import configure
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from apps.chat.tests.test_rag_source_figures import figure_for
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse

pytestmark = [pytest.mark.asyncio, pytest.mark.django_db(transaction=True)]


@pytest.mark.parametrize(
    "kind,mutation",
    [
        ("whole", "none"),
        ("whole", "figure"),
        ("whole", "revoke"),
        ("adjacent", "edit"),
        ("adjacent", "none"),
    ],
)
async def test_normal_current_tool_handoff(docs, monkeypatch, kind, mutation):
    from apps.chat.services.rag_turn import preservation_turn
    from apps.chat.services.tool_wiring.documents import (
        more_context_tool,
        whole_document_tool,
    )
    from apps.collections.models import CollectionPermission
    from apps.documents.models import DocumentFigure, TextChunk
    from lib.llm.providers.image_context import serialize_tool_result_for_llm
    from lib.llm.providers.openai import OpenAIInterface

    configure(monkeypatch)
    monkeypatch.setenv("RAG_EVIDENCE_TOKEN_BUDGET", "10000")
    user, doc, chunks, _ = docs
    figure = (
        await database_sync_to_async(figure_for)(
            user, doc, "Original caption: only below 1 Pa."
        )
        if kind == "whole"
        else None
    )
    calls = []
    provider = OpenAIInterface(None, "test")

    async def get_message(**kwargs):
        calls.append(kwargs)
        return LLMResponse(
            text=f"Only below 1 Pa [doc:{doc.id} chunk:{chunks[0].pk}].",
            tool_call={},
            stop_reason="stop",
            input_usage=0,
            output_usage=0,
            model="test",
        )

    monkeypatch.setattr(provider, "get_message", get_message)
    convo = Conversation(
        system="sys",
        messages=[UserMessage(content="Show the document and its figures")],
    )
    consumer = SimpleNamespace(
        user=user,
        col_ref=CollectionsRef([doc.collection_id]),
        convo=convo,
        llm_if=provider,
    )
    async with preservation_turn(consumer, max_func_calls=4) as runtime:
        if kind == "whole":
            tool = whole_document_tool(
                user, SimpleNamespace(chat=consumer), consumer.col_ref
            )
            arguments, name = {"doc_id": str(doc.id)}, "whole_document"
        else:
            tool = more_context_tool(user)
            arguments, name = (
                {"chunk_id": chunks[1].pk, "adjacent_chunks": 1},
                "more_context",
            )
        raw = await database_sync_to_async(tool, thread_sensitive=False)(**arguments)
        if mutation == "revoke":
            await database_sync_to_async(
                lambda: CollectionPermission.objects.filter(user=user).delete()
            )()
        if mutation == "edit":
            await database_sync_to_async(
                lambda: TextChunk.objects.filter(pk=chunks[0].pk).update(
                    content="changed"
                )
            )()
        if mutation == "figure":
            await database_sync_to_async(
                lambda: DocumentFigure.objects.filter(id=figure.id).update(
                    extracted_caption="Changed caption"
                )
            )()
        working = convo + [
            ToolMessage(
                tool_name=name,
                for_whom="assistant",
                arguments=arguments,
                result_dict=raw,
                content=serialize_tool_result_for_llm(raw),
            )
        ]
        result, _ = await provider.complete(working, 2048)
        assert runtime.budget.actions_used == 1
    if mutation in {"revoke", "edit"}:
        assert calls == []
        assert "limits" in result[-1].content
    else:
        assert len(calls) == 1
        text = str(calls[0]["messages"])
        assert ("Original caption" in text) is (kind == "whole" and mutation == "none")
        assert "Changed caption" not in text


async def test_expired_normal_tool_result_cannot_restart_legacy_loop(docs, monkeypatch):
    from apps.chat.services.rag_turn import preservation_turn
    from apps.chat.services.tool_wiring.documents import vector_search_tool

    configure(monkeypatch)
    user, doc, _, _ = docs
    consumer = SimpleNamespace(user=user, col_ref=CollectionsRef([doc.collection_id]))
    async with preservation_turn(consumer, max_func_calls=1) as runtime:
        runtime.budget.reserve_action("already spent")
        raw = await database_sync_to_async(
            vector_search_tool(user, consumer.col_ref), thread_sensitive=False
        )(search_string="again", top_k=5)
        assert raw["retrieval_status"] == "context_limited"
        assert runtime.budget.actions_used == 1
