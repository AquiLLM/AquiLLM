"""Tests for the direct RAG pipeline orchestrator (Tasks 4 and 5)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

from apps.chat.refs import CollectionsRef
from apps.chat.services import rag_pipeline
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
from lib.llm.types.response import LLMResponse


def _user_convo(text: str) -> Conversation:
    return Conversation(system="sys", messages=[UserMessage(content=text)])


def _consumer(convo: Conversation, collections: list) -> SimpleNamespace:
    return SimpleNamespace(
        user=object(),
        col_ref=CollectionsRef(list(collections)),
        convo=convo,
        _send_stream_payload=AsyncMock(),
    )


def _results_payload() -> dict:
    return {
        "result": [
            {
                "rank": 1,
                "chunk_id": 1,
                "doc_id": "doc-a",
                "title": "Paper A",
                "text": "Calibration uses flat fields and dark frames.",
                "citation": "[doc:doc-a chunk:1]",
            }
        ],
        "retrieval_status": "results_found",
        "retrieved_count": 1,
        "retrieved_documents": ["Paper A"],
    }


async def test_skipped_when_flag_off(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "0")
    convo = _user_convo("search the selected documents for calibration")
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "skipped"
    llm_if.get_message.assert_not_called()


async def test_skipped_when_intent_not_rag(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    convo = _user_convo("hello there, how are you")
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "skipped"
    llm_if.get_message.assert_not_called()


async def test_skipped_when_last_message_not_user(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="hi"),
            AssistantMessage(content="hello", stop_reason="end_turn"),
        ],
    )
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "skipped"


async def test_handled_retrieves_before_llm(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    order: list[str] = []
    raw = _results_payload()

    def fake_search(consumer, query, top_k):
        order.append("retrieval")
        return raw

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        order.append("synthesis")
        return convo + [
            AssistantMessage(content="Answer [doc:doc-a chunk:1].", stop_reason="end_turn")
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("search the selected documents for calibration")
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert order == ["retrieval", "synthesis"]
    llm_if.get_message.assert_not_called()
    assert "Answer" in consumer.convo[-1].content


async def test_handled_appends_synthetic_tool_messages(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    raw = _results_payload()

    captured: dict = {}

    def fake_search(consumer, query, top_k):
        return raw

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        captured["convo"] = convo
        captured["packet"] = packet
        return convo + [AssistantMessage(content="done", stop_reason="end_turn")]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("search the documents for calibration")
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    synth_convo = captured["convo"]
    # user -> assistant(tool_call) -> tool(result)
    assert isinstance(synth_convo[-1], ToolMessage)
    assert synth_convo[-1].for_whom == "assistant"
    assert synth_convo[-1].tool_name == "vector_search"
    assert isinstance(synth_convo[-2], AssistantMessage)
    assert synth_convo[-2].tool_call_name == "vector_search"
    assert captured["packet"].chunks


async def test_no_collections_prompts_selection(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    called: list = []
    monkeypatch.setattr(
        rag_pipeline, "_run_vector_search", lambda *a, **k: called.append(1)
    )

    convo = _user_convo("search the documents for calibration")
    consumer = _consumer(convo, [])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert not called
    last = consumer.convo[-1]
    assert isinstance(last, AssistantMessage)
    assert "collection" in last.content.lower()


async def test_retrieval_failure_falls_back_to_skipped(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")

    def boom(consumer, query, top_k):
        raise RuntimeError("retrieval backend down")

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", boom)

    convo = _user_convo("search the documents for calibration")
    consumer = _consumer(convo, [1])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "skipped"
    # consumer.convo must be untouched so the normal tool loop can run.
    assert consumer.convo is convo
    assert isinstance(consumer.convo[-1], UserMessage)


async def test_end_to_end_real_synthesis_single_llm_call(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_ENFORCE_CHUNK_CITATIONS", "1")
    order: list[str] = []
    raw = _results_payload()

    def fake_search(consumer, query, top_k):
        order.append("retrieval")
        return raw

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)

    answer = (
        "The paper describes a calibration method using flat fields and dark "
        "frames to remove instrument signatures [doc:doc-a chunk:1]."
    )
    llm_if = _FakeLLMInterface(
        [
            LLMResponse(
                text=answer,
                tool_call=None,
                stop_reason="end_turn",
                input_usage=1,
                output_usage=1,
            )
        ]
    )

    # Track that the first LLM call happens only after retrieval.
    original_get_message = llm_if.get_message

    async def tracked_get_message(*args, **kwargs):
        order.append("get_message")
        return await original_get_message(*args, **kwargs)

    llm_if.get_message = tracked_get_message

    convo = _user_convo("search the selected documents for the calibration method")
    consumer = _consumer(convo, [1])

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert order[0] == "retrieval"
    assert order.count("get_message") == 1
    assert "calibration" in consumer.convo[-1].content.lower()
    assert "[doc:doc-a chunk:1]" in consumer.convo[-1].content


async def test_direct_rag_no_results_returns_notice_without_llm(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    raw = {
        "result": [],
        "retrieval_status": "no_results",
        "retrieval_message": (
            'I searched the selected documents for "dark matter", '
            "but retrieval returned no relevant passages."
        ),
    }
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda c, q, k: raw)

    convo = _user_convo("search the documents for dark matter")
    consumer = _consumer(convo, [1])
    # Empty response list: any LLM call would raise IndexError.
    llm_if = _FakeLLMInterface([])

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert llm_if.calls == []
    assert "no relevant passages" in consumer.convo[-1].content.lower()


async def test_direct_rag_figure_request_embeds_image(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    raw = _results_payload()
    raw["result"][0]["image_url"] = "/aquillm/document_image/doc-a/"
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda c, q, k: raw)

    answer = (
        "The figure shows calibration drift across magnitude bins for the survey "
        "sample [doc:doc-a chunk:1]."
    )
    llm_if = _FakeLLMInterface(
        [
            LLMResponse(
                text=answer,
                tool_call=None,
                stop_reason="end_turn",
                input_usage=1,
                output_usage=1,
            )
        ]
    )

    convo = _user_convo("show me the figure for the calibration method")
    consumer = _consumer(convo, [1])

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert "/aquillm/document_image/doc-a/" in consumer.convo[-1].content
