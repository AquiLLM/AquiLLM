"""Manual retrieval admits visual payloads only for the resolved visual request."""

from types import SimpleNamespace

import pytest

from apps.chat.refs import CollectionsRef
from apps.chat.services import manual_search_turn, rag_manual_preservation
from apps.chat.services.manual_search_commands import (
    parse_manual_search,
    resolve_search_query,
)
from apps.chat.services.tool_wiring import documents
from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from apps.documents.services.source_loading import (
    SourceRuntime,
    current_source_runtime,
    source_runtime_scope,
)
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

DOC = "11111111-1111-4111-8111-111111111111"
CASES = [
    ("Show Figure 2 for calibration", True),
    ("Explain calibration", False),
    ("Explain Figure 2 in text only, without images", False),
]


def _result(document_id, include_images):
    row = {
        "doc_id": document_id,
        "chunk_id": 1,
        "text": "The calibration plot shows instrument drift.",
        "citation": f"[doc:{document_id} chunk:1]",
        "type": "text",
    }
    if include_images:
        row.update(
            type="text_with_image",
            image_url=f"/aquillm/document_image/{document_id}/",
        )
    return {"result": [row], "retrieved_count": 1}


def _conversation(command, query, inherit):
    messages = [UserMessage(content=query)] if inherit else []
    messages.append(UserMessage(content=command if inherit else f"{command} {query}"))
    return Conversation(system="sys", messages=messages)


@pytest.mark.parametrize("query,visual", CASES)
@pytest.mark.parametrize("inherit", [False, True])
@pytest.mark.parametrize("single_document", [False, True])
async def test_legacy_manual_search_admits_only_requested_images(
    monkeypatch, query, visual, inherit, single_document
):
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "legacy")
    monkeypatch.setenv("RAG_RERANK_TEXT_MODE", "legacy")
    monkeypatch.setenv("RAG_DOCUMENT_CAPACITY_MODE", "legacy")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "0")
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "0")
    command = f"/search [{DOC}]" if single_document else "/collection"
    convo = _conversation(command, query, inherit)
    consumer = SimpleNamespace(user=object(), col_ref=CollectionsRef([1]), convo=convo)

    def search(*, search_string, top_k, include_images=False, **kwargs):
        assert search_string == query
        assert 1 <= top_k <= 15
        assert kwargs == ({"doc_id": DOC} if single_document else {})
        return _result(DOC, include_images)

    tool_name = "search_single_document" if single_document else "vector_search"
    monkeypatch.setattr(
        manual_search_turn, tool_name + "_tool", lambda user, cols: search
    )
    reply = f"The calibration plot shows drift [doc:{DOC} chunk:1]."
    llm = _FakeLLMInterface(
        [
            LLMResponse(
                text=reply,
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
            )
        ]
    )

    status = await manual_search_turn.run_manual_search_turn(consumer, llm, convo)

    assert status == "handled"
    retrieved = consumer.convo[-2]
    assert isinstance(retrieved, ToolMessage)
    assert retrieved.tool_name == tool_name
    assert bool(retrieved.arguments.get("include_images")) is visual
    assert ("image_url" in retrieved.result_dict["result"][0]) is visual
    assert consumer.convo[-1].content == reply
    assert "![" not in consumer.convo[-1].content


@pytest.mark.parametrize("query,visual", CASES)
@pytest.mark.parametrize("inherit", [False, True])
@pytest.mark.parametrize("single_document", [False, True])
def test_preserved_manual_search_keeps_scope_and_visual_intent(
    docs, monkeypatch, query, visual, inherit, single_document
):
    user, doc, _, auth = docs
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    document_id = str(doc.id)
    prefix = f"/search [{document_id}]" if single_document else "/collection"
    convo = _conversation(prefix, query, inherit)
    command = parse_manual_search(convo[-1].content)
    resolved = resolve_search_query(command, convo.messages[:-1])
    consumer = SimpleNamespace(
        user=user, col_ref=CollectionsRef([doc.collection_id]), convo=convo
    )
    parent = SourceRuntime(TurnBudget(TurnLimits()), auth)

    def search(*, search_string, top_k, include_images=False, **kwargs):
        assert search_string == query
        assert 1 <= top_k <= 15
        assert kwargs == ({"doc_id": document_id} if single_document else {})
        runtime = current_source_runtime()
        assert runtime.budget is parent.budget
        assert runtime.authorization.selected_document_ids == frozenset((doc.id,))
        return _result(document_id, include_images)

    tool_name = "search_single_document" if single_document else "vector_search"
    monkeypatch.setattr(documents, tool_name + "_tool", lambda user, cols: search)

    with source_runtime_scope(parent):
        raw, acquired_scope = rag_manual_preservation._manual_acquire(
            consumer, command, resolved, convo.messages[:-1]
        )
        assert current_source_runtime() is parent

    assert ("image_url" in raw["result"][0]) is visual
    assert acquired_scope.budget is parent.budget
    assert acquired_scope.cache is parent.cache
    assert acquired_scope.authorization.selected_document_ids == frozenset((doc.id,))


@pytest.mark.parametrize("prefix", ["/collection", f"/search [{DOC}]"])
@pytest.mark.parametrize("query,visual", CASES)
async def test_source_manual_history_preserves_image_choice_for_retry(
    monkeypatch,
    prefix,
    query,
    visual,
):
    from apps.chat.services.rag_pipeline_messages import _retry_needs_visual_tools

    convo = _conversation(prefix, query, inherit=True)
    consumer = SimpleNamespace(user=object(), col_ref=CollectionsRef([1]), convo=convo)
    runtime = SourceRuntime(
        TurnBudget(TurnLimits()), SimpleNamespace(selected_collection_ids=(1,))
    )
    monkeypatch.setattr(
        rag_manual_preservation,
        "_manual_acquire",
        lambda *args: (_result(DOC, visual), runtime),
    )

    async def finish(consumer, llm, working, **kwargs):
        return working + [AssistantMessage(content="Answer", stop_reason="end_turn")]

    monkeypatch.setattr(rag_manual_preservation, "finish_preservation", finish)
    with source_runtime_scope(runtime):
        status = await rag_manual_preservation.run_manual_preservation(
            consumer,
            object(),
            convo,
            parse_manual_search(prefix),
            stream_func=None,
        )
    assert status == "handled"
    tool_message = consumer.convo[-2]
    assert isinstance(tool_message, ToolMessage)
    assert bool(tool_message.arguments.get("include_images")) is visual
    retried = consumer.convo + [UserMessage(content="try again")]
    assert _retry_needs_visual_tools(retried) is visual
