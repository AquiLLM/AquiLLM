"""Actual direct source selection with an injected shared turn runtime."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apps.chat.refs import CollectionsRef
from apps.chat.tests.test_rag_selection_scoring import (
    Policy,
    _authorization,
    _chunk,
    _pool,
)
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_direct_source_route_keeps_five_passages_and_tail(monkeypatch):
    from apps.chat.services import rag_pipeline, rag_source_hydration
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
    chunks = tuple(
        _chunk(i, f"Necessary aspect {i}. " * 20 + "Tail condition: only below 1 Pa.")
        for i in range(1, 6)
    )
    pool = _pool(chunks)
    for row in pool.rows:
        row["text"] = "clipped public preview"
    monkeypatch.setattr(
        rag_pipeline, "_run_vector_search", lambda *_: {"result": list(pool.rows)}
    )
    monkeypatch.setattr(
        rag_source_hydration, "source_query_rows", lambda *_a, **_kw: chunks
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
    consumer = SimpleNamespace(user=object(), col_ref=CollectionsRef([1]), convo=convo)
    runtime = SourceRuntime(TurnBudget(TurnLimits()), _authorization(Policy()))
    with source_runtime_scope(runtime):
        assert (
            await rag_pipeline.run_direct_rag_turn(
                consumer, SimpleNamespace(complete=complete), convo
            )
            == "handled"
        )
    assert len(captured[0]["result"]) == 5
    assert [r["text"] for r in captured[0]["result"]] == [c.content for c in chunks]


@pytest.mark.asyncio
async def test_missing_ledger_source_route_is_handled_limited_not_fallback(monkeypatch):
    from apps.chat.services import rag_pipeline

    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda *_: {"result": []})
    convo = Conversation(
        system="sys", messages=[UserMessage(content="Summarize the selected documents")]
    )
    consumer = SimpleNamespace(user=object(), col_ref=CollectionsRef([1]), convo=convo)
    llm = SimpleNamespace(complete=AsyncMock())
    assert await rag_pipeline.run_direct_rag_turn(consumer, llm, convo) == "handled"
    assert "limits" in consumer.convo[-1].content
    llm.complete.assert_not_called()
