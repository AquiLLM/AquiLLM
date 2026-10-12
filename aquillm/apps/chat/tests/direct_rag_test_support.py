"""Tests for the direct RAG pipeline orchestrator (Tasks 4 and 5)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from apps.chat.refs import CollectionsRef
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage


def _user_convo(text: str) -> Conversation:
    return Conversation(system="sys", messages=[UserMessage(content=text)])


async def _consumer(convo: Conversation, collections: list, *, rows=None):
    from apps.chat.tests.persisted_direct_test_support import persisted_direct_case
    from lib.llm.types.messages import ToolMessage
    declared = list(_results_payload()["result"] if rows is None else rows)
    for message in convo.messages:
        if isinstance(message, ToolMessage) and isinstance(message.result_dict, dict):
            declared.extend(message.result_dict.get("result", ()))
    case = await persisted_direct_case(declared, collection_labels=tuple(collections) or (1,))
    for message in convo.messages:
        if isinstance(message, ToolMessage) and isinstance(message.result_dict, dict):
            message.result_dict = case.rows(message.result_dict)
        if isinstance(message.content, str):
            message.content = case.text(message.content)
    consumer = case.consumer(convo, collections)
    consumer.case = case
    consumer.test_synthesis_calls = []
    return consumer


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



async def _synthesize(consumer, convo, packet, answer, *, stream_func=None):
    """Real synthesis/publication, with only the external SDK transport synthetic."""
    from apps.chat.services.rag_synthesis import synthesize_from_evidence
    from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
    from lib.llm.types.response import LLMResponse
    transport = _FakeLLMInterface([LLMResponse(text=consumer.case.text(answer),
        tool_call=None, stop_reason="end_turn", input_usage=1, output_usage=1)])
    result = await synthesize_from_evidence(transport, convo, packet, stream_func=stream_func)
    assert len(transport.calls) == 1
    consumer.test_synthesis_calls.extend(transport.calls)
    return result



def _admit_query(query):
    from apps.chat.services.rag_action_tools import admit_tool_action
    assert admit_tool_action("vector", query)


def _retrieval(consumer, query, payload):
    _admit_query(query)
    return consumer.case.rows(payload)


def _assert_default_excerpt(consumer, *, draft):
    message = consumer.convo[-1]
    citation = consumer.case.text("[doc:doc-a chunk:1]")
    assert message.content == (
        "Support for an answer is uncertain in the selected evidence. "
        "These are attributed excerpts, not a conclusion about the question:\n\n"
        f"> Calibration uses flat fields and dark frames. {citation}"
    )
    assert draft not in message.content
    assert message._validated_answer.outcome == "partial"
    assert len(consumer.test_synthesis_calls) == 1
