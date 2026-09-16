"""Manual commands execute exactly the requested retrieval before synthesis."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apps.chat.refs import CollectionsRef
from apps.chat.services import manual_search_turn, rag_pipeline
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

DOC = "11111111-1111-4111-8111-111111111111"


def _consumer(text, collections=(1,), prior=()):
    convo = Conversation(system="sys", messages=[*prior, UserMessage(content=text)])
    return SimpleNamespace(
        user=object(), col_ref=CollectionsRef(list(collections)), convo=convo
    )


def _payload():
    return {
        "result": [
            {
                "doc_id": DOC,
                "chunk_id": 1,
                "text": "Calibration evidence.",
                "citation": f"[doc:{DOC} chunk:1]",
            }
        ],
        "retrieved_count": 1,
    }


@pytest.mark.parametrize(
    "text,tool_name,expected",
    [
        (
            "/collection flat field calibration",
            "vector_search",
            {"search_string": "flat field calibration"},
        ),
        (
            f"/search [{DOC}] calibration",
            "search_single_document",
            {"doc_id": DOC, "search_string": "calibration"},
        ),
    ],
)
async def test_manual_command_overrides_disabled_auto_rag_and_local_tool_heuristics(
    monkeypatch,
    text,
    tool_name,
    expected,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "0")
    calls = []

    def factory(user, col_ref):
        def search(**kwargs):
            calls.append((user, col_ref, kwargs))
            return _payload()

        return search

    monkeypatch.setattr(manual_search_turn, tool_name + "_tool", factory)
    synth = AsyncMock(
        side_effect=lambda llm, convo, packet, **kwargs: (
            convo
            + [AssistantMessage(content="Grounded answer", stop_reason="end_turn")]
        )
    )
    monkeypatch.setattr(manual_search_turn, "synthesize_from_evidence", synth)
    consumer = _consumer(text)
    llm = SimpleNamespace(get_message=AsyncMock())
    assert (
        await rag_pipeline.run_direct_rag_turn(consumer, llm, consumer.convo)
        == "handled"
    )
    assert len(calls) == 1
    assert calls[0][:2] == (consumer.user, consumer.col_ref)
    assert {key: calls[0][2][key] for key in expected} == expected
    assert 1 <= calls[0][2]["top_k"] <= 15
    assert consumer.convo[-3].tool_call_name == tool_name
    assert isinstance(consumer.convo[-2], ToolMessage)
    assert consumer.convo[-2].arguments == calls[0][2]
    assert consumer.convo[-1].content == "Grounded answer"
    llm.get_message.assert_not_called()


async def test_named_document_search_loads_only_authorized_selected_documents(
    monkeypatch,
):
    consumer = _consumer("/search [Paper A] calibration")
    docs = [SimpleNamespace(id=DOC, title="Paper A")]
    calls = []
    monkeypatch.setattr(
        manual_search_turn,
        "load_search_documents",
        lambda c: docs if c is consumer else [],
    )
    monkeypatch.setattr(
        manual_search_turn,
        "search_single_document_tool",
        lambda user, cols: lambda **kw: calls.append(kw) or {"result": []},
    )
    assert (
        await rag_pipeline.run_direct_rag_turn(consumer, object(), consumer.convo)
        == "handled"
    )
    assert calls[0]["doc_id"] == DOC
    assert "no relevant passages" in consumer.convo[-1].content


@pytest.mark.parametrize(
    "text,collections,notice",
    [
        ("/collection query", (), "collection"),
        ("/collection", (1,), "question"),
        ("/search query", (1,), "document"),
        ("/search [broken", (1,), "/search"),
    ],
)
async def test_missing_scope_or_bad_input_is_handled_without_llm(
    monkeypatch, text, collections, notice
):
    monkeypatch.setattr(manual_search_turn, "load_search_documents", lambda c: [])
    consumer = _consumer(text, collections)
    llm = SimpleNamespace(get_message=AsyncMock())
    assert (
        await rag_pipeline.run_direct_rag_turn(consumer, llm, consumer.convo)
        == "handled"
    )
    assert notice in consumer.convo[-1].content
    llm.get_message.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("private backend detail"), {"exception": "private access detail"}],
)
async def test_manual_search_failure_never_falls_back_to_model_routing(
    monkeypatch, failure
):
    def search(**kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(
        manual_search_turn, "search_single_document_tool", lambda *args: search
    )
    consumer = _consumer(f"/search {DOC} query", collections=())
    assert (
        await rag_pipeline.run_direct_rag_turn(consumer, object(), consumer.convo)
        == "handled"
    )
    assert "private" not in consumer.convo[-1].content
    assert "couldn't" in consumer.convo[-1].content


async def test_synthesis_failure_preserves_retrieved_evidence_without_retrying_search(
    monkeypatch,
):
    monkeypatch.setattr(
        manual_search_turn,
        "vector_search_tool",
        lambda *args: lambda **kwargs: _payload(),
    )
    monkeypatch.setattr(
        manual_search_turn,
        "synthesize_from_evidence",
        AsyncMock(side_effect=RuntimeError("private")),
    )
    consumer = _consumer("/collection calibration")
    assert (
        await rag_pipeline.run_direct_rag_turn(consumer, object(), consumer.convo)
        == "handled"
    )
    assert isinstance(consumer.convo[-2], ToolMessage)
    assert consumer.convo[-2].result_dict == _payload()
    assert "private" not in consumer.convo[-1].content
