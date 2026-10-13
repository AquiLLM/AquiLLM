"""Actual direct source selection with an injected shared turn runtime."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from channels.db import database_sync_to_async
from django.db import connections

from apps.chat.refs import CollectionsRef
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest_asyncio.fixture(autouse=True)
async def close_publication_worker(transactional_db):
    yield
    await database_sync_to_async(connections.close_all)()


@pytest.fixture
def source_owner(transactional_db):
    from django.contrib.auth.models import User
    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
    user = User.objects.create_user(username="source-pipeline-owner")
    collection = Collection.objects.create(name="Source pipeline")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    bodies = [f"Necessary aspect {i}. " * 20 + "Tail condition: only below 1 Pa." for i in range(1, 6)]
    document = RawTextDocument(title="Five aspects", full_text="\n".join(bodies),
                               collection=collection, ingested_by=user)
    document.save(dont_rechunk=True)
    rows, start = [], 0
    for i, text in enumerate(bodies):
        rows.append(TextChunk(doc_id=document.id, content=text, chunk_number=i,
                              start_position=start, end_position=start + len(text)))
        start += len(text) + 1
    chunks = tuple(TextChunk.objects.bulk_create(rows))
    authority = build_selected_scope_authorization_context(principal=user,
        selected_collection_ids=(collection.pk,), selected_documents=(document,))
    return user, document, chunks, authority


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_direct_source_route_keeps_five_passages_and_tail(source_owner, monkeypatch):
    from apps.chat.services import rag_pipeline
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_runtime_scope,
    )
    from lib.llm.types.messages import AssistantMessage

    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setenv("RAG_DOCUMENT_CAPACITY_MODE", "budgeted")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    user, document, chunks, authority = source_owner
    rows = [dict(doc_id=str(document.id), chunk_id=chunk.pk, chunk=chunk.chunk_number,
                 title=document.title, text="clipped public preview",
                 citation=f"[doc:{document.id} chunk:{chunk.pk}]") for chunk in chunks]
    monkeypatch.setattr(
        rag_pipeline, "_run_vector_search", lambda *_: {"result": rows}
    )
    captured = []

    async def complete(request, *_args, **_kw):
        captured.append(request[-1].result_dict)
        return request + [
            AssistantMessage(content="Supported response", stop_reason="end_turn")
        ], "changed"

    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Summarize the five aspects in the selected documents")
        ],
    )
    consumer = SimpleNamespace(user=user, col_ref=CollectionsRef([document.collection_id]), convo=convo)
    assert (
        await rag_pipeline.run_direct_rag_turn(
            consumer, SimpleNamespace(complete=complete), convo
        )
        == "handled"
    )
    assert len(captured[0]["result"]) == 5
    assert [r["text"] for r in captured[0]["result"]] == [c.content for c in chunks]


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_missing_ledger_source_route_is_handled_limited_not_fallback(source_owner, monkeypatch):
    from apps.chat.services import rag_pipeline
    from lib.llm.turn_context import TurnContext, bind_turn

    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda *_: {"result": []})
    convo = Conversation(
        system="sys", messages=[UserMessage(content="Summarize the selected documents")]
    )
    # A selected real owner does not mint a missing source ledger inside an
    # already-started turn. Ordinary empty-picker routing is tested separately.
    consumer = SimpleNamespace(user=source_owner[0],
        col_ref=CollectionsRef([source_owner[1].collection_id]), convo=convo)
    llm = SimpleNamespace(complete=AsyncMock())
    from lib.llm.answer_publication import LIMITED_ANSWER
    with bind_turn(TurnContext(TurnBudget(TurnLimits()))):
        assert await rag_pipeline.run_direct_rag_turn(consumer, llm, convo) == "handled"
    assert consumer.convo[-1].content == LIMITED_ANSWER
    llm.complete.assert_not_called()
