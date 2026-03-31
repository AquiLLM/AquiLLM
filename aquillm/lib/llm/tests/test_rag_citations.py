"""Tests for strict, verifiable RAG citation handling."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.providers.rag_citations import (
    collect_allowed_chunk_citations,
    find_invalid_citations,
    response_has_required_citations,
    synthesize_cited_extract_from_results,
)
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse


def test_collect_allowed_chunk_citations_from_verbose_and_compact_rows():
    convo = Conversation(
        system="sys",
        messages=[
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {"chunk_id": 7, "doc_id": "doc-a", "title": "Doc A", "text": "alpha"},
                        {"i": 9, "d": "doc-b", "n": "Doc B", "x": "beta"},
                    ]
                },
            )
        ],
    )
    allowed = collect_allowed_chunk_citations(convo)
    assert "[doc:doc-a chunk:7]" in allowed
    assert "[doc:doc-b chunk:9]" in allowed


def test_collect_allowed_chunk_citations_accepts_explicit_ref_fields():
    convo = Conversation(
        system="sys",
        messages=[
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {"citation": "[doc:doc-c chunk:11]", "text": "gamma"},
                        {"ref": "prefix [doc:doc-d chunk:12] suffix", "x": "delta"},
                    ]
                },
            )
        ],
    )
    allowed = collect_allowed_chunk_citations(convo)
    assert "[doc:doc-c chunk:11]" in allowed
    assert "[doc:doc-d chunk:12]" in allowed


def test_response_has_required_citations_and_rejects_unknown_refs():
    allowed = {"[doc:doc-a chunk:7]"}
    assert response_has_required_citations("Fact from source [doc:doc-a chunk:7].", allowed)
    assert not response_has_required_citations("Fact with no cite.", allowed)
    invalid = find_invalid_citations(
        "Wrong cite [doc:doc-a chunk:999] plus right [doc:doc-a chunk:7].",
        allowed,
    )
    assert invalid == ["[doc:doc-a chunk:999]"]


def test_response_has_required_citations_rejects_uncited_bullets():
    allowed = {"[doc:doc-a chunk:7]"}
    answer = "- Supported claim [doc:doc-a chunk:7]\n- Unsupported claim without cite"
    assert not response_has_required_citations(answer, allowed)


def test_synthesize_cited_extract_from_results_includes_refs():
    convo = Conversation(
        system="sys",
        messages=[
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {
                            "chunk_id": 7,
                            "doc_id": "doc-a",
                            "title": "Doc A",
                            "text": "Alpha finding with supporting detail.",
                        }
                    ]
                },
            )
        ],
    )
    fallback = synthesize_cited_extract_from_results(convo)
    assert fallback is not None
    assert "Alpha finding" in fallback
    assert "[doc:doc-a chunk:7]" in fallback


@pytest.mark.asyncio
async def test_complete_turn_retries_when_post_tool_answer_lacks_required_citations():
    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            side_effect=[
                LLMResponse(
                    text="This answer forgot citations.",
                    tool_call={},
                    stop_reason="stop",
                    input_usage=1,
                    output_usage=1,
                    model="fake",
                ),
                LLMResponse(
                    text="Cited answer [doc:doc-a chunk:7].",
                    tool_call={},
                    stop_reason="stop",
                    input_usage=1,
                    output_usage=1,
                    model="fake",
                ),
            ]
        ),
    )
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="What does the source say?"),
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {
                            "chunk_id": 7,
                            "doc_id": "doc-a",
                            "title": "Doc A",
                            "text": "Alpha finding with supporting detail.",
                        }
                    ]
                },
            ),
        ],
    )
    updated, changed = await complete_conversation_turn(llm, convo, max_tokens=512)
    assert changed == "changed"
    assert updated[-1].content == "Cited answer [doc:doc-a chunk:7]."
    assert llm.get_message.await_count == 2
