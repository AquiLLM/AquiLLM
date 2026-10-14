"""Actual adapters, shared selector and provider-neutral completion handoff."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apps.chat.refs import CollectionsRef
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage

pytestmark = [pytest.mark.asyncio, pytest.mark.django_db(transaction=True)]


def configure(monkeypatch, source=True, capacity=False, windowed=False):
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if source else "legacy")
    monkeypatch.setenv(
        "RAG_DOCUMENT_CAPACITY_MODE", "budgeted" if capacity else "legacy"
    )
    monkeypatch.setenv("RAG_RERANK_TEXT_MODE", "windowed" if windowed else "legacy")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "3")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "32000")
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")


def fake_search(monkeypatch):
    from apps.documents.models import TextChunk
    from apps.documents.services.source_loading import source_query_rows

    def search(query, top_k, documents, **kwargs):
        rows = source_query_rows(
            TextChunk.objects.filter(doc_id__in=[d.id for d in documents]).order_by(
                "pk"
            )
        )
        return None, None, rows, {}

    monkeypatch.setattr(TextChunk, "text_chunk_search", search)


@pytest.mark.parametrize(
    "source,capacity,windowed",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
async def test_direct_real_route_mode_matrix_one_action_one_selector(
    docs, monkeypatch, source, capacity, windowed
):
    from apps.chat.services import rag_pipeline, rag_preservation_turn
    from apps.chat.services.rag_turn import preservation_turn

    user, doc, chunks, _ = docs
    configure(monkeypatch, source, capacity, windowed)
    fake_search(monkeypatch)
    selections, requests = [], []
    original = rag_preservation_turn.prepare_selection_turn

    def select(*args, **kwargs):
        selections.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(rag_pipeline, "_prepare_selection_sync", select)

    async def complete(request, *args, **kwargs):
        requests.append(request)
        return request + [
            AssistantMessage(content="Qualified evidence", stop_reason="end_turn")
        ], "changed"

    convo = Conversation(
        system="sys", messages=[UserMessage(content="Summarize the selected documents")]
    )
    consumer = SimpleNamespace(
        user=user, col_ref=CollectionsRef([doc.collection_id]), convo=convo
    )
    async with preservation_turn(consumer, max_func_calls=4) as runtime:
        outcome = await rag_pipeline.run_direct_rag_turn(
            consumer, SimpleNamespace(complete=complete), convo
        )
        assert runtime.budget.actions_used == 1
        assert runtime.budget.sources_used == len(chunks)
    assert outcome == "handled"
    assert len(selections) == len(requests) == 1
    assert chunks[-1].content in requests[0][-1].content


async def test_manual_exact_document_outside_selection_uses_same_budget(
    docs, monkeypatch
):
    from apps.chat.services.manual_search_turn import run_manual_search_turn
    from apps.chat.services.rag_turn import preservation_turn
    from lib.llm.turn_context import current_turn

    user, doc, chunks, _ = docs
    configure(monkeypatch)
    fake_search(monkeypatch)
    captured = []

    async def complete(request, *args, **kwargs):
        captured.append(request)
        return request + [
            AssistantMessage(content="Supported", stop_reason="end_turn")
        ], "changed"

    convo = Conversation(
        system="sys", messages=[UserMessage(content=f"/search {doc.id} summarize")]
    )
    consumer = SimpleNamespace(user=user, col_ref=CollectionsRef([]), convo=convo)
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        assert runtime is None  # Empty picker has no authorized public source scope.
        original_budget = current_turn().budget
        assert (
            await run_manual_search_turn(
                consumer, SimpleNamespace(complete=complete), convo
            )
            == "handled"
        )
        assert current_turn().budget is original_budget and original_budget.actions_used == 1
    assert len(captured) == 1
    assert chunks[-1].content in captured[0][-1].content
    assert consumer.col_ref.collections == []


@pytest.mark.parametrize("iterative", [False, True])
async def test_normal_model_tool_is_counted_once_and_synthesized_once(
    docs, monkeypatch, iterative
):
    from apps.chat.services.rag_turn import preservation_turn
    from apps.chat.services.tool_wiring.documents import vector_search_tool
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.types.response import LLMResponse
    from lib.llm.types.tools import ToolChoice

    user, doc, chunks, _ = docs
    configure(monkeypatch)
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", str(int(iterative)))
    fake_search(monkeypatch)
    calls = []
    provider = OpenAIInterface(None, "test")

    async def get_message(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return LLMResponse(
                text=None,
                tool_call={
                    "tool_call_id": "one",
                    "tool_call_name": "vector_search",
                    "tool_call_input": {"search_string": "aspects", "top_k": 5},
                },
                stop_reason="tool_use",
                input_usage=0,
                output_usage=0,
                model="test",
            )
        if kwargs["max_tokens"] == 512:
            return LLMResponse(
                text=(
                    '{"requested_aspects":["exception"],"support":[],"unreso'
                    'lved_aspects":["exception"],"next_action":{"kind":"vect'
                    'or","query":"exception","aspect":"exception"}}'
                ),
                tool_call={},
                stop_reason="stop",
                input_usage=0,
                output_usage=0,
                model="test",
            )
        return LLMResponse(
            text=f"Only below 1 Pa [doc:{doc.id} chunk:{chunks[0].pk}].",
            tool_call={},
            stop_reason="stop",
            input_usage=0,
            output_usage=0,
            model="test",
        )

    monkeypatch.setattr(provider, "get_message", get_message)
    ref = CollectionsRef([doc.collection_id])
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(
                content="Summarize selected documents",
                tools=[vector_search_tool(user, ref)],
                tool_choice=ToolChoice(type="auto"),
            )
        ],
    )
    consumer = SimpleNamespace(user=user, col_ref=ref, convo=convo)
    send = AsyncMock()
    async with preservation_turn(consumer, max_func_calls=4) as runtime:
        await provider.spin(convo, 4, send, 2048)
        assert runtime.budget.actions_used == 1
    assert len(calls) == (3 if iterative else 2)
    assert not calls[-1].get("tools")
    assert chunks[-1].content in str(calls[-1]["messages"])
    published = send.call_args.args[0][-1]
    assert "uncertain" in published.content
    for chunk in chunks:
        citation = f"[doc:{doc.id} chunk:{chunk.pk}]"
        assert (
            f"> Aspect {chunk.chunk_number}. {citation} "
            f"Qualified tail: only below 1 Pa. {citation}"
        ) in published.content
    assert published._validated_answer.outcome == "partial"


async def test_direct_native_off_search_reaches_preserved_completion(docs, monkeypatch):
    from django.conf import settings

    from apps.chat.services import rag_pipeline
    from apps.chat.services.rag_turn import preservation_turn
    from apps.documents.services import chunk_rerank_selection_provider, chunk_search
    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )
    from apps.documents.tests.test_chunk_rerank_window_scores import Scorer

    user, doc, chunks, _ = docs
    configure(monkeypatch, capacity=True, windowed=True)
    monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", "native_closure")
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "adaptive")
    monkeypatch.setattr(settings, "KG_OVERLAY_ENABLED", False)
    monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "RAG_RRF_ENABLED", False, raising=False)
    monkeypatch.setattr(chunk_search, "load_query_embedding", lambda query: None)
    monkeypatch.setattr(
        chunk_rerank_selection_provider, "current_selection_scorer",
        lambda *, deadline, clock, turn_budget: WindowSelectionScorer(
            Scorer(), budget=turn_budget, deadline=deadline, clock=clock,
            pair_counter=lambda q, d: len(q) + len(d) + 8,
        ),
    )
    stages, requests = [], []
    original = chunk_search.retrieval_stages

    def observe(*args, **kwargs):
        stages.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(chunk_search, "retrieval_stages", observe)

    async def complete(request, *args, **kwargs):
        requests.append(request)
        return request + [
            AssistantMessage(content="Qualified evidence", stop_reason="end_turn")
        ], "changed"

    convo = Conversation(system="sys", messages=[UserMessage(content="Qualified tail")])
    consumer = SimpleNamespace(
        user=user, col_ref=CollectionsRef([doc.collection_id]), convo=convo
    )
    async with preservation_turn(consumer, max_func_calls=4) as runtime:
        outcome = await rag_pipeline.run_direct_rag_turn(
            consumer, SimpleNamespace(complete=complete), convo
        )
        assert runtime.budget.pairs_used["acquisition"] == 0
        assert 0 < runtime.budget.pairs_used["final"] <= 45
        assert runtime.budget.limits.retrieval_ms == 15000
    assert outcome == "handled" and len(requests) == 1 and len(stages) == 1
    assert stages[0]["acquisition_schedule_observed"] == "native_closure"
    assert chunks[-1].content in requests[0][-1].content
    assert f"[doc:{doc.id} chunk:{chunks[-1].pk}]" in requests[0][-1].content
