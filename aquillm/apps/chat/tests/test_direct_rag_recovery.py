"""Direct-RAG routing, recovery, and observability backport regressions."""
from json import dumps
from types import SimpleNamespace
from unittest.mock import AsyncMock

from apps.chat.consumers.chat_transport import best_effort_send
from apps.chat.services import rag_metrics, rag_pipeline
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from apps.chat.tests.test_direct_rag_pipeline import _consumer, _results_payload, _user_convo
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage


async def test_explicit_collection_synthesis_without_selection_is_handled(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    convo = _user_convo(
        "Hi aquillm can you tell me about the documents in this collection and synthesize things?"
    )
    consumer = _consumer(convo, [])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    llm_if.get_message.assert_not_called()
    assert "no collections are selected" in consumer.convo[-1].content.lower()


async def test_stream_disconnect_does_not_turn_direct_rag_into_tool_fallback(
    monkeypatch,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    convo = _user_convo("what is attensity")
    consumer = _consumer(convo, [203])
    consumer.transport_connected = True

    async def raw_send(*, text_data):
        raise RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'"
        )

    consumer.send = raw_send

    async def safe_stream(payload):
        await best_effort_send(
            consumer,
            text_data=dumps({"stream": payload}),
        )

    async def fake_synth(_llm_if, working_convo, _packet, *, stream_func=None):
        await stream_func({"content": "partial", "done": False})
        return working_convo + [
            AssistantMessage(
                content="Grounded answer [doc:doc-a chunk:1]",
                stop_reason="end_turn",
            )
        ]

    monkeypatch.setattr(
        rag_pipeline,
        "_run_vector_search",
        lambda *_args: _results_payload(),
    )
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(),
        convo,
        stream_func=safe_stream,
    )

    assert outcome == "handled"
    assert consumer.transport_connected is False
    assert consumer.convo[-1].content == "Grounded answer [doc:doc-a chunk:1]"


async def test_selected_collection_definition_uses_direct_rag(monkeypatch):
    """A terse selected-collection question must skip model tool selection."""
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    searched_queries: list[str] = []

    def fake_search(_consumer, query, _top_k):
        searched_queries.append(query)
        return _results_payload()

    async def fake_synth(_llm_if, working_convo, _packet, *, stream_func=None):
        return working_convo + [
            AssistantMessage(
                content="Attensity answer [doc:doc-a chunk:1].",
                stop_reason="end_turn",
            )
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("what is attensity")
    consumer = _consumer(convo, [203])
    llm_if = SimpleNamespace(get_message=AsyncMock())

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert searched_queries == ["what is attensity"]
    llm_if.get_message.assert_not_called()


async def test_retry_reuses_last_direct_vector_query(monkeypatch):
    """Retrying a direct retrieval reuses its resolved query, not the word retry."""
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    searched_queries: list[str] = []

    def fake_search(_consumer, query, _top_k):
        searched_queries.append(query)
        return _results_payload()

    async def fake_synth(_llm_if, working_convo, _packet, *, stream_func=None):
        return working_convo + [
            AssistantMessage(
                content="Retried answer [doc:doc-a chunk:1].",
                stop_reason="end_turn",
            )
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="what is attensity"),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="call-1",
                tool_call_name="vector_search",
                tool_call_input={"search_string": "attensity", "top_k": 10},
            ),
            ToolMessage(
                tool_name="vector_search",
                for_whom="assistant",
                content="evidence",
                arguments={"search_string": "attensity", "top_k": 10},
                result_dict=_results_payload(),
            ),
            AssistantMessage(content="Earlier answer.", stop_reason="end_turn"),
            UserMessage(content="retry"),
        ],
    )
    consumer = _consumer(convo, [203])

    outcome = await run_direct_rag_turn(
        consumer,
        SimpleNamespace(get_message=AsyncMock()),
        convo,
        stream_func=None,
    )

    assert outcome == "handled"
    assert searched_queries == ["attensity"]


async def test_multi_part_direct_rag_searches_variants_before_one_synthesis(
    monkeypatch,
):
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
        return convo + [
            AssistantMessage(content="Cited answer", stop_reason="end_turn")
        ]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", fake_synth)

    convo = _user_convo("Explain what each paper is about? What overlaps between them?")
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

    convo = _user_convo("Explain what each paper is about? What overlaps between them?")
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


def test_direct_rag_metrics_record_safe_stage_fields(monkeypatch):
    captured: dict = {}

    def capture(event, **fields):
        captured["event"] = event
        captured.update(fields)

    monkeypatch.setattr(rag_metrics.logger, "info", capture)

    rag_metrics.log_direct_rag_turn(
        correlation_id="0f22db7309f04ab0a4676cdb5a76f962",
        intent_ms=1.1,
        query_ms=2.2,
        retrieval_ms=3.3,
        evidence_ms=4.4,
        synthesis_ms=5.5,
        persistence_ms=0.2,
        total_ms=16.5,
        retrieved_count=2,
        retained_count=2,
        retrieval_status="results_found",
    )

    assert captured["event"] == "rag_direct_turn"
    assert captured["correlation_id"] == "0f22db7309f04ab0a4676cdb5a76f962"
    assert captured["retained_count"] == 2
    assert captured["persistence_ms"] == 0.2
