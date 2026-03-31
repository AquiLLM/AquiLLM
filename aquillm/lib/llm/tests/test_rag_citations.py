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


def test_response_has_required_citations_allows_uncited_connective_sentence():
    allowed = {"[doc:doc-a chunk:7]"}
    answer = (
        "- Core finding from the paper [doc:doc-a chunk:7]\n"
        "Overall, this suggests the pattern is robust across experiments."
    )
    assert response_has_required_citations(answer, allowed)


def test_response_has_required_citations_allows_indented_sub_bullets_under_cited_heading():
    allowed = {"[doc:doc-a chunk:7]"}
    answer = (
        "Figure 27 - Degrees of wrinkledness [doc:doc-a chunk:7]\n"
        "    - Telegraph seeds 1-4\n"
        "    - Telephone seeds 5-6\n"
        "    - Lightning seeds 13-15"
    )
    assert response_has_required_citations(answer, allowed)


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


@pytest.mark.asyncio
async def test_complete_turn_uses_higher_default_post_tool_token_budget(monkeypatch):
    monkeypatch.delenv("LLM_POST_TOOL_MAX_TOKENS", raising=False)
    seen_max_tokens: list[int] = []

    async def _fake_get_message(**kwargs):
        seen_max_tokens.append(int(kwargs.get("max_tokens", 0)))
        return LLMResponse(
            text="Cited answer [doc:doc-a chunk:7].",
            tool_call={},
            stop_reason="stop",
            input_usage=1,
            output_usage=1,
            model="fake",
        )

    llm = SimpleNamespace(base_args={}, get_message=AsyncMock(side_effect=_fake_get_message))
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
    _updated, changed = await complete_conversation_turn(llm, convo, max_tokens=2048)
    assert changed == "changed"
    assert seen_max_tokens
    assert seen_max_tokens[0] == 1536


@pytest.mark.asyncio
async def test_complete_turn_truncates_long_prior_answer_in_citation_retry(monkeypatch):
    monkeypatch.setenv("LLM_CITATION_RETRY_PRIOR_MAX_CHARS", "120")
    captured_retry_prompt: dict[str, str] = {}

    async def _fake_get_message(**kwargs):
        messages = kwargs.get("messages") or []
        if len(messages) >= 2:
            maybe_retry = messages[-1].get("content") if isinstance(messages[-1], dict) else ""
            if isinstance(maybe_retry, str) and "Allow-list:" in maybe_retry:
                captured_retry_prompt["text"] = maybe_retry
                return LLMResponse(
                    text="Cited answer [doc:doc-a chunk:7].",
                    tool_call={},
                    stop_reason="stop",
                    input_usage=1,
                    output_usage=1,
                    model="fake",
                )
        return LLMResponse(
            text=("Very long uncited draft sentence. " * 200).strip(),
            tool_call={},
            stop_reason="stop",
            input_usage=1,
            output_usage=1,
            model="fake",
        )

    llm = SimpleNamespace(base_args={}, get_message=AsyncMock(side_effect=_fake_get_message))
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Summarize with citations."),
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
    _updated, changed = await complete_conversation_turn(llm, convo, max_tokens=2048)
    assert changed == "changed"
    prompt_text = captured_retry_prompt.get("text", "")
    assert "[Truncated for citation retry.]" in prompt_text


@pytest.mark.asyncio
async def test_complete_turn_soft_accepts_high_quality_without_invalid_citations():
    long_answer = (
        "- The book tracks how Mendelism became politically loaded in public debate [doc:doc-a chunk:7]\n"
        "- Radick emphasizes that scientific authority was shaped by institutions and networks\n"
        "- Statistical and biometric traditions repeatedly intersected with heredity arguments [doc:doc-a chunk:7]\n"
        "- The narrative ties laboratory claims to broader social consequences in Britain [doc:doc-a chunk:7]\n"
    )
    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            return_value=LLMResponse(
                text=long_answer,
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
                model="fake",
            )
        ),
    )
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Summarize with citations."),
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={"result": [{"chunk_id": 7, "doc_id": "doc-a", "text": "alpha"}]},
            ),
        ],
    )
    updated, changed = await complete_conversation_turn(llm, convo, max_tokens=2048)
    assert changed == "changed"
    assert updated[-1].content == long_answer
    assert llm.get_message.await_count == 1


@pytest.mark.asyncio
async def test_complete_turn_does_not_force_extractive_fallback_for_incomplete_but_non_fabricated_citations():
    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            side_effect=[
                LLMResponse(
                    text="- One cited point [doc:doc-a chunk:7]\n- One uncited point",
                    tool_call={},
                    stop_reason="stop",
                    input_usage=1,
                    output_usage=1,
                    model="fake",
                ),
                LLMResponse(
                    text="- Revised cited point [doc:doc-a chunk:7]\n- Still uncited point",
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
            UserMessage(content="Summarize with citations."),
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={"result": [{"chunk_id": 7, "doc_id": "doc-a", "text": "alpha"}]},
            ),
        ],
    )
    updated, changed = await complete_conversation_turn(llm, convo, max_tokens=1024)
    assert changed == "changed"
    assert "I can only provide claims directly supported by retrieved chunks" not in updated[-1].content
    assert llm.get_message.await_count == 2


@pytest.mark.asyncio
async def test_complete_turn_streaming_skips_citation_rewrite_retry():
    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            return_value=LLMResponse(
                text="- One cited point [doc:doc-a chunk:7]\n- One uncited point",
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
                model="fake",
            )
        ),
    )
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Summarize with citations."),
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={"result": [{"chunk_id": 7, "doc_id": "doc-a", "text": "alpha"}]},
            ),
        ],
    )

    async def _noop_stream(_payload: dict):
        return None

    updated, changed = await complete_conversation_turn(
        llm,
        convo,
        max_tokens=1024,
        stream_func=_noop_stream,
    )
    assert changed == "changed"
    assert llm.get_message.await_count == 1
    assert "One uncited point" in updated[-1].content


@pytest.mark.asyncio
async def test_complete_turn_streaming_invalid_citations_get_note_not_full_fallback():
    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            return_value=LLMResponse(
                text="Claim with fabricated cite [doc:doc-a chunk:999].",
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
                model="fake",
            )
        ),
    )
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Summarize with citations."),
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={"result": [{"chunk_id": 7, "doc_id": "doc-a", "text": "alpha"}]},
            ),
        ],
    )

    async def _noop_stream(_payload: dict):
        return None

    updated, changed = await complete_conversation_turn(
        llm,
        convo,
        max_tokens=1024,
        stream_func=_noop_stream,
    )
    assert changed == "changed"
    assert llm.get_message.await_count == 1
    assert "could not be verified" in updated[-1].content
    assert "I can only provide claims directly supported by retrieved chunks" not in updated[-1].content


@pytest.mark.asyncio
async def test_complete_turn_buffers_stream_until_citations_finalize():
    stream_payloads: list[dict] = []

    async def _capture_stream(payload: dict):
        stream_payloads.append(payload)

    async def _fake_get_message(**kwargs):
        assert kwargs.get("stream_callback") is None
        return LLMResponse(
            text="Final cited answer [doc:doc-a chunk:7].",
            tool_call={},
            stop_reason="stop",
            input_usage=5,
            output_usage=7,
            model="fake",
        )

    llm = SimpleNamespace(base_args={}, get_message=AsyncMock(side_effect=_fake_get_message))
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Summarize with citations."),
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={"result": [{"chunk_id": 7, "doc_id": "doc-a", "text": "alpha"}]},
            ),
        ],
    )

    updated, changed = await complete_conversation_turn(
        llm,
        convo,
        max_tokens=1024,
        stream_func=_capture_stream,
    )
    assert changed == "changed"
    assert llm.get_message.await_count == 1
    assert len(stream_payloads) == 1
    assert stream_payloads[0]["done"] is True
    assert stream_payloads[0]["content"] == "Final cited answer [doc:doc-a chunk:7]."
    assert stream_payloads[0]["usage"] == 12
    assert updated[-1].content == "Final cited answer [doc:doc-a chunk:7]."
