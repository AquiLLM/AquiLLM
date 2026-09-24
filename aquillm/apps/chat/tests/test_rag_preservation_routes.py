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
        assert (
            await run_manual_search_turn(
                consumer, SimpleNamespace(complete=complete), convo
            )
            == "handled"
        )
        assert runtime.budget.actions_used == 1
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
    assert send.call_args.args[0][-1].content.startswith("Only below")


@pytest.mark.parametrize("route", ["direct", "normal"])
async def test_partial_followup_keeps_available_source_without_substitution(
    docs, monkeypatch, route
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
            UserMessage(content="Compare both documents"),
        ],
    )
    monkeypatch.setattr(
        TextChunk,
        "text_chunk_search",
        lambda *a, **kw: pytest.fail(
            "missing requested source must not be substituted"
        ),
    )
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
            from apps.chat.services.rag_preservation_turn import normal_source_handoff

            runtime.budget.reserve_action("model selected initial action")
            working = convo + [
                ToolMessage(
                    tool_name="vector_search",
                    for_whom="assistant",
                    content="empty",
                    arguments={},
                    result_dict={"result": []},
                )
            ]
            await normal_source_handoff(
                consumer, SimpleNamespace(complete=complete), working, 2048, None
            )
        assert runtime.budget.actions_used == (route == "normal")
        assert runtime.budget.sources_used == 1
    assert len(requests) == 1
    assert chunks[0].content in requests[0][-1].content
    assert missing not in requests[0][-1].content
    assert "no longer access" in requests[0][-1].content
