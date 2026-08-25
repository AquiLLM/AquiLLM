"""Tests for the direct RAG pipeline orchestrator (Tasks 4 and 5)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from apps.chat.refs import CollectionsRef
from apps.chat.services import rag_metrics, rag_pipeline
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
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


async def test_multi_part_direct_rag_searches_variants_before_one_synthesis(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "3")
    searched: list[str] = []
    synthesized: list = []

    def fake_search(consumer, query, top_k):
        searched.append(query)
        chunk_id = len(searched)
        return {
            "result": [
                {
                    "rank": 1,
                    "chunk_id": chunk_id,
                    "doc_id": f"doc-{chunk_id}",
                    "title": f"Paper {chunk_id}",
                    "text": f"Evidence for query {chunk_id}.",
                    "citation": f"[doc:doc-{chunk_id} chunk:{chunk_id}]",
                }
            ],
            "retrieval_status": "results_found",
            "retrieved_count": 1,
            "retrieved_documents": [f"Paper {chunk_id}"],
        }

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        synthesized.append(packet)
        return convo + [AssistantMessage(content="Cited answer", stop_reason="end_turn")]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo(
        "Explain what each paper is about? What overlaps between them?"
    )
    consumer = _consumer(convo, [1, 2, 3])

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(get_message=AsyncMock()),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    assert searched == [
        "Explain what each paper is about? What overlaps between them?",
        "Explain what each paper is about",
        "What overlaps between them",
    ]
    assert len(synthesized) == 1
    assert len(synthesized[0].citation_tokens) == 3


async def test_multi_query_retrieval_uses_successful_variants_when_one_fails(
    monkeypatch,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "3")
    synthesized: list = []

    def fake_search(consumer, query, top_k):
        if query == "What overlaps between them":
            raise RuntimeError("one query backend failure")
        return _results_payload()

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        synthesized.append(packet)
        return convo + [
            AssistantMessage(content="Cited answer", stop_reason="end_turn")
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo(
        "Explain what each paper is about? What overlaps between them?"
    )
    consumer = _consumer(convo, [1, 2, 3])

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(get_message=AsyncMock()),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    assert len(synthesized) == 1
    assert synthesized[0].citation_tokens == ["[doc:doc-a chunk:1]"]


async def test_graph_overlay_failure_keeps_direct_rag_on_vector_evidence(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    raw = _results_payload()
    # Task 16 fails graph work open to this unchanged baseline payload.
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda *_args: raw)

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        assert packet.citation_tokens == ["[doc:doc-a chunk:1]"]
        return convo + [
            AssistantMessage(
                content="Vector evidence remains usable [doc:doc-a chunk:1].",
                stop_reason="end_turn",
            )
        ]

    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)
    convo = _user_convo("search the selected documents for calibration")
    consumer = _consumer(convo, [1])

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(get_message=AsyncMock()),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    assert "Vector evidence remains usable" in consumer.convo[-1].content


async def test_direct_rag_logs_safe_graph_contribution(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    raw = _results_payload()
    raw["_retrieval_diagnostics"] = {
        "graph_status": "hit",
        "graph_ms": 4.5,
        "graph_seed_count": 3,
        "graph_candidate_count": 2,
        "graph_path": ["private-node"],
    }
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda *_args: raw)

    async def fake_synth(llm_if, convo, packet, *, stream_func=None):
        return convo + [AssistantMessage(content="Cited answer", stop_reason="end_turn")]

    captured: dict = {}

    def capture_metrics(**fields):
        captured.update(fields)

    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)
    monkeypatch.setattr(rag_pipeline, "log_direct_rag_turn", capture_metrics)
    convo = _user_convo("search the selected documents for calibration")

    outcome = await run_direct_rag_turn(
        _consumer(convo, [1]),
        SimpleNamespace(get_message=AsyncMock()),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    assert captured["graph_status"] == "hit"
    assert captured["graph_candidate_count"] == 2
    assert "private-node" not in repr(captured)


def test_direct_rag_metrics_accept_safe_optional_graph_fields(monkeypatch):
    captured: dict = {}

    def capture(event, **fields):
        captured["event"] = event
        captured.update(fields)

    monkeypatch.setattr(rag_metrics.logger, "info", capture)

    rag_metrics.log_direct_rag_turn(
        intent_ms=1.1,
        query_ms=2.2,
        retrieval_ms=3.3,
        evidence_ms=4.4,
        synthesis_ms=5.5,
        total_ms=16.5,
        retrieved_count=2,
        retrieval_status="results_found",
        graph_ms=0.7,
        graph_seed_count=3,
        graph_candidate_count=1,
        graph_status="hit",
        graph_algorithm_signature="a" * 64,
        graph_version_signature="b" * 64,
    )

    assert captured["event"] == "rag_direct_turn"
    assert captured["graph_status"] == "hit"
    assert captured["graph_seed_count"] == 3
    assert captured["graph_candidate_count"] == 1
    assert captured["graph_algorithm_signature"] == "a" * 64
    assert captured["graph_version_signature"] == "b" * 64


def test_direct_rag_metrics_omit_poisoned_graph_fields(monkeypatch):
    captured: dict = {}

    def capture(event, **fields):
        captured["event"] = event
        captured.update(fields)

    monkeypatch.setattr(rag_metrics.logger, "info", capture)

    rag_metrics.log_direct_rag_turn(
        intent_ms=1.0,
        query_ms=1.0,
        retrieval_ms=1.0,
        evidence_ms=1.0,
        synthesis_ms=1.0,
        total_ms=5.0,
        retrieved_count=1,
        retrieval_status="results_found",
        graph_ms=float("nan"),
        graph_seed_count=True,
        graph_candidate_count=21,
        graph_status="query=private text",
        graph_algorithm_signature="PRIVATE-LABEL" * 6,
        graph_version_signature="A" * 64,
    )

    assert captured["event"] == "rag_direct_turn"
    assert not any(key.startswith("graph_") for key in captured)


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
    assert "thinking_budget" not in llm_if.calls[0]
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
        "retrieval_diagnostics": {
            "doc_count": 1,
            "vector_error": None,
        },
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
