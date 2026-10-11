"""Normal whole/adjacent handoff revalidates current chunk and figure authority."""

from types import SimpleNamespace

import pytest
from channels.db import database_sync_to_async

from apps.chat.refs import CollectionsRef
from apps.chat.tests.test_rag_preservation_routes import configure
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from apps.chat.tests.test_rag_source_figures import figure_for
from apps.documents.tests.representation_support import upload as upload
from apps.documents.tests.test_representation_media import figure as figure
from .manual_routing_support import current_figure as current_figure, close_routing_worker as close_routing_worker
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
        ("image", "none"),
        ("image", "text"),
        ("image", "revoke"),
        ("adjacent", "edit"),
        ("adjacent", "none"),
    ],
)
async def test_normal_current_tool_handoff(docs, current_figure, monkeypatch, kind, mutation):
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
    if kind == "image":
        user, doc, selected_chunk = current_figure
        chunks = [selected_chunk]
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
        if kind in {"whole", "image"}:
            tool = whole_document_tool(
                user, SimpleNamespace(chat=consumer), consumer.col_ref
            )
            arguments, name = (
                {"doc_id": str(doc.id), "include_images": mutation != "text"},
                "whole_document",
            )
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
            def edit_owner():
                doc.full_text = "Changed canonical source"
                doc.save(dont_rechunk=True)
            await database_sync_to_async(edit_owner)()
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
        from lib.llm.answer_publication import LIMITED_ANSWER
        assert result[-1].content == LIMITED_ANSWER
    else:
        assert len(calls) == 1
        text = str(calls[0]["messages"])
        # The whole-parent packet has no selected figure chunk; its unbound
        # auxiliary is omitted without discarding the parent's cited text.
        assert "Original caption" not in text
        assert "Changed caption" not in text
        if kind == "image":
            assert (f"/aquillm/document_image/{doc.id}/" in text) is (
                mutation == "none"
            )


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
