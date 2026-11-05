"""Tests for strict, verifiable RAG citation handling."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.providers.rag_citations import (
    synthesize_cited_extract_from_results,
)
from lib.llm.providers.request_observability import observability_scope
from lib.llm.tests.publication_fixture_support import (
    ALPHA_EXCERPT_FALLBACK,
    assert_alpha_excerpt_fallback,
    selected_publication,
)
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse


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
    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(llm, convo, max_tokens=512)
    assert changed == "changed"
    assert updated[-1].content == "Cited answer [doc:doc-a chunk:7]."
    assert updated[-1]._validated_answer.outcome == "partial"
    assert llm.get_message.await_count == 2

@pytest.mark.asyncio
async def test_complete_turn_uses_higher_default_post_tool_token_budget(monkeypatch):
    monkeypatch.delenv("LLM_POST_TOOL_MAX_TOKENS", raising=False)
    monkeypatch.setenv("LLM_POST_TOOL_OUTPUT_MAX_TOKENS", "6144")
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
    assert seen_max_tokens[0] == 6144


@pytest.mark.asyncio
async def test_direct_synthesis_honors_its_explicit_completion_budget(monkeypatch):
    monkeypatch.setenv("LLM_POST_TOOL_OUTPUT_MAX_TOKENS", "12288")
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

    with observability_scope("0f22db7309f04ab0a4676cdb5a76f962", "direct_synthesis"):
        _updated, changed = await complete_conversation_turn(
            llm,
            convo,
            max_tokens=4096,
        )

    assert changed == "changed"
    assert seen_max_tokens == [4096]


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
async def test_complete_turn_rejects_long_answer_with_uncited_factual_line():
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
    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(llm, convo, max_tokens=2048)
    assert changed == "changed"
    assert_alpha_excerpt_fallback(updated[-1])
    assert llm.get_message.await_count == 2
    assert all(call.kwargs["stream_callback"] is None for call in llm.get_message.await_args_list)


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
async def test_complete_turn_final_only_stream_still_repairs_uncited_claims(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
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

    events = []

    async def _noop_stream(payload: dict):
        events.append(payload)

    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(
            llm,
            convo,
            max_tokens=1024,
            stream_func=_noop_stream,
        )
    assert changed == "changed"
    assert_alpha_excerpt_fallback(updated[-1])
    assert llm.get_message.await_count == 2
    assert all(call.kwargs["stream_callback"] is None for call in llm.get_message.await_args_list)
    assert len(events) == 1 and events[0]["done"] is True
    assert events[0]["content"] == ALPHA_EXCERPT_FALLBACK


@pytest.mark.asyncio
async def test_complete_turn_final_only_stream_rejects_fabricated_citations(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
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

    events = []

    async def _noop_stream(payload: dict):
        events.append(payload)

    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(
            llm,
            convo,
            max_tokens=1024,
            stream_func=_noop_stream,
        )
    assert changed == "changed"
    assert_alpha_excerpt_fallback(updated[-1])
    assert llm.get_message.await_count == 2
    assert all(call.kwargs["stream_callback"] is None for call in llm.get_message.await_args_list)
    assert len(events) == 1 and events[0]["done"] is True
    assert events[0]["content"] == ALPHA_EXCERPT_FALLBACK


@pytest.mark.asyncio
async def test_complete_turn_legacy_stream_flag_publishes_only_final_attribution(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
    stream_payloads: list[dict] = []

    async def _capture_stream(payload: dict):
        stream_payloads.append(payload)

    async def _fake_get_message(**kwargs):
        cb = kwargs.get("stream_callback")
        assert cb is None
        return LLMResponse(
            text="Final answer [doc:doc-a chunk:7].",
            tool_call={},
            stop_reason="stop",
            input_usage=4,
            output_usage=5,
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

    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(
            llm,
            convo,
            max_tokens=1024,
            stream_func=_capture_stream,
        )
    assert changed == "changed"
    assert llm.get_message.await_count == 1
    expected = "Final answer [doc:doc-a chunk:7].\n\nSources:\n- [doc:doc-a chunk:7]"
    assert updated[-1].content == expected
    assert len(stream_payloads) == 1
    assert stream_payloads[0]["done"] is True
    assert stream_payloads[0]["content"] == expected
    assert "Partial answer" not in expected


@pytest.mark.asyncio
async def test_complete_turn_streaming_skips_appending_sources_when_flag_disabled(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
    monkeypatch.setenv("RAG_APPEND_CITATION_SOURCES", "0")
    stream_payloads: list[dict] = []

    async def _capture_stream(payload: dict):
        stream_payloads.append(payload)

    async def _fake_get_message(**kwargs):
        cb = kwargs.get("stream_callback")
        assert cb is None
        return LLMResponse(
            text="Narrative summary with no inline citations.",
            tool_call={},
            stop_reason="stop",
            input_usage=2,
            output_usage=3,
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

    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(
            llm,
            convo,
            max_tokens=1024,
            stream_func=_capture_stream,
        )
    assert changed == "changed"
    assert_alpha_excerpt_fallback(updated[-1])
    assert len(stream_payloads) == 1 and stream_payloads[0]["done"] is True
    assert stream_payloads[0]["content"] == ALPHA_EXCERPT_FALLBACK
    assert llm.get_message.await_count == 2
    assert all(call.kwargs["stream_callback"] is None for call in llm.get_message.await_args_list)


@pytest.mark.asyncio
async def test_complete_turn_sources_cannot_rescue_uncited_model_answer(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
    stream_payloads: list[dict] = []

    async def _capture_stream(payload: dict):
        stream_payloads.append(payload)

    async def _fake_get_message(**kwargs):
        cb = kwargs.get("stream_callback")
        assert cb is None
        return LLMResponse(
            text="Narrative summary with no inline citations.",
            tool_call={},
            stop_reason="stop",
            input_usage=2,
            output_usage=3,
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

    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(
            llm,
            convo,
            max_tokens=1024,
            stream_func=_capture_stream,
        )
    assert changed == "changed"
    assert_alpha_excerpt_fallback(updated[-1])
    assert len(stream_payloads) == 1 and stream_payloads[0]["done"] is True
    assert stream_payloads[0]["content"] == ALPHA_EXCERPT_FALLBACK
    assert llm.get_message.await_count == 2
    assert all(call.kwargs["stream_callback"] is None for call in llm.get_message.await_args_list)


@pytest.mark.asyncio
async def test_complete_turn_final_stream_adds_refs_when_model_emits_empty_sources_heading(monkeypatch):
    monkeypatch.setenv("RAG_ENFORCE_CHUNK_CITATIONS", "1")
    stream_payloads: list[dict] = []

    async def _capture_stream(payload: dict):
        stream_payloads.append(payload)

    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            return_value=LLMResponse(
                text=(
                    "The retrieved passage supports the central claim [doc:doc-a chunk:7].\n\n"
                    "Sources:"
                ),
                tool_call={},
                stop_reason="stop",
                input_usage=2,
                output_usage=3,
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

    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(
            llm,
            convo,
            max_tokens=1024,
            stream_func=_capture_stream,
        )

    assert changed == "changed"
    from lib.llm.providers.complete_turn import _append_citation_sources_if_missing
    assert _append_citation_sources_if_missing(
        "A cited fact [doc:doc-a chunk:7].\n\nSources:", {"[doc:doc-a chunk:7]"}
    ).endswith("Sources:\n- [doc:doc-a chunk:7]")
    expected = (
        "The retrieved passage supports the central claim [doc:doc-a chunk:7]."
        "\n\nSources:\n- [doc:doc-a chunk:7]"
    )
    assert updated[-1].content == expected
    assert len(stream_payloads) == 1 and stream_payloads[0]["done"] is True
    assert stream_payloads[0]["content"] == expected


@pytest.mark.asyncio
async def test_complete_turn_streaming_appends_used_sources_when_present(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
    stream_payloads: list[dict] = []

    async def _capture_stream(payload: dict):
        stream_payloads.append(payload)

    async def _fake_get_message(**kwargs):
        cb = kwargs.get("stream_callback")
        assert cb is None
        return LLMResponse(
            text="Summary with one inline cite [doc:doc-a chunk:1].",
            tool_call={},
            stop_reason="stop",
            input_usage=2,
            output_usage=3,
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
                        {"chunk_id": 1, "doc_id": "doc-a", "text": "a"},
                        {"chunk_id": 2, "doc_id": "doc-b", "text": "b"},
                        {"chunk_id": 3, "doc_id": "doc-c", "text": "c"},
                        {"chunk_id": 4, "doc_id": "doc-d", "text": "d"},
                        {"chunk_id": 5, "doc_id": "doc-e", "text": "e"},
                        {"chunk_id": 6, "doc_id": "doc-f", "text": "f"},
                        {"chunk_id": 7, "doc_id": "doc-g", "text": "g"},
                    ]
                },
            ),
        ],
    )

    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(
            llm,
            convo,
            max_tokens=1024,
            stream_func=_capture_stream,
        )
    assert changed == "changed"
    assert llm.get_message.await_count == 1
    assert len(stream_payloads) == 1
    streamed = stream_payloads[0]["content"]
    from lib.llm.providers.complete_turn import _append_citation_sources_if_missing
    refs = {f"[doc:doc-{letter} chunk:{index}]" for index, letter in enumerate("abcdefg", 1)}
    source_list = _append_citation_sources_if_missing("Summary [doc:doc-a chunk:1].", refs)
    assert "Sources:\n- [doc:doc-a chunk:1]" in source_list
    assert "[doc:doc-g chunk:7]" not in source_list
    expected = (
        "Summary with one inline cite [doc:doc-a chunk:1]."
        "\n\nSources:\n- [doc:doc-a chunk:1]"
    )
    assert streamed == updated[-1].content == expected
    assert "[doc:doc-f" not in streamed and "[doc:doc-g" not in streamed


@pytest.mark.asyncio
async def test_complete_turn_cutoff_draft_cannot_escape_final_validation(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
    stream_payloads: list[dict] = []

    async def _capture_stream(payload: dict):
        stream_payloads.append(payload)

    async def _fake_get_message(**kwargs):
        cb = kwargs.get("stream_callback")
        assert cb is None
        return LLMResponse(
            text="![fig. 6: 1), based at the recently established University (Page 168) Source: Gre",
            tool_call={},
            stop_reason="max_tokens",
            input_usage=2,
            output_usage=2,
            model="fake",
        )

    async def _fake_continue_cutoff_response(**_kwargs):
        return LLMResponse(
            text="en and then completed [doc:doc-a chunk:7].",
            tool_call={},
            stop_reason="stop",
            input_usage=1,
            output_usage=1,
            model="fake",
        )

    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(side_effect=_fake_get_message),
        _continue_cutoff_response=AsyncMock(side_effect=_fake_continue_cutoff_response),
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

    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(
            llm,
            convo,
            max_tokens=1024,
            stream_func=_capture_stream,
        )
    assert changed == "changed"
    expected = LIMITED_ANSWER
    assert updated[-1].content == expected
    assert updated[-1]._validated_answer.outcome == "limited"
    assert len(stream_payloads) == 1 and stream_payloads[0]["done"] is True
    assert stream_payloads[0]["content"] == expected
    llm._continue_cutoff_response.assert_awaited_once()
    assert llm._continue_cutoff_response.await_args.kwargs["stream_callback"] is None


@pytest.mark.asyncio
async def test_complete_turn_deduplicated_continuation_preserves_cited_generated_answer(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            return_value=LLMResponse(
                text=(
                    "# The GalaxiesML Dataset: Comprehensive Technical Analysis\n"
                    "Dataset Construction and Provenance\n"
                    "Source Survey: HSC-PDR2 Wide Survey\n"
                    "Performance Requirements Context\n"
                    "The paper establishes **LS"
                ),
                tool_call={},
                stop_reason="max_tokens",
                input_usage=2,
                output_usage=2,
                model="fake",
            )
        ),
        _continue_cutoff_response=AsyncMock(
            return_value=LLMResponse(
                text=(
                    "# The GalaxiesML Dataset: Comprehensive Technical Analysis\n"
                    "Dataset Construction and Provenance\n"
                    "Source Survey: HSC-PDR2 Wide Survey\n"
                    "Performance Requirements Context\n"
                    "The paper establishes **LSST deployment thresholds and acceptance criteria [doc:doc-a chunk:7]."
                ),
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

    from lib.llm.providers.complete_turn_continuation import (
        _trim_duplicate_continuation_prefix,
    )
    partial = llm.get_message.return_value.text
    continuation = llm._continue_cutoff_response.return_value.text
    merged = partial + _trim_duplicate_continuation_prefix(partial, continuation)
    assert merged.count("# The GalaxiesML Dataset: Comprehensive Technical Analysis") == 1
    assert "LSST deployment thresholds" in merged
    stream = AsyncMock()
    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(
            llm,
            convo,
            max_tokens=1024,
            stream_func=stream,
        )
    assert changed == "changed"
    expected = (
        "# The GalaxiesML Dataset: Comprehensive Technical Analysis\n"
        "Dataset Construction and Provenance\n"
        "Source Survey: HSC-PDR2 Wide Survey\n"
        "Performance Requirements Context\n"
        "The paper establishes **LSST deployment thresholds and acceptance criteria "
        "[doc:doc-a chunk:7].\n\nSources:\n- [doc:doc-a chunk:7]"
    )
    assert updated[-1].content == expected
    assert updated[-1]._validated_answer.outcome == "partial"
    llm._continue_cutoff_response.assert_awaited_once()
    assert llm._continue_cutoff_response.await_args.kwargs["stream_callback"] is None
    stream.assert_awaited_once()
    assert stream.await_args.args[0]["done"] is True
    assert stream.await_args.args[0]["content"] == expected


@pytest.mark.asyncio
async def test_complete_turn_unbound_whole_document_continuation_stays_limited(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
    events = []

    async def _noop_stream(payload: dict):
        events.append(payload)

    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            return_value=LLMResponse(
                text=(
                    "Figure 1 from TurboQuant Paper\n\n"
                    "Context and Analysis\n"
                    "This figure compares the distribution characteristics"
                ),
                tool_call={},
                stop_reason="max_tokens",
                input_usage=2,
                output_usage=2,
                model="fake",
            )
        ),
        _continue_cutoff_response=AsyncMock(
            return_value=LLMResponse(
                text=(
                    " across bit-width settings and highlights the bias-variance tradeoff in retrieval."
                ),
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
            UserMessage(content="Show me figure 1 and explain it."),
            ToolMessage(
                content="{}",
                tool_name="whole_document",
                for_whom="assistant",
                arguments={"doc_id": "doc-whole-123"},
                result_dict={
                    "result": {
                        "type": "image_document",
                        "text": "full text",
                        "image_url": "/aquillm/document_image/doc-whole-123/",
                    }
                },
            ),
        ],
    )

    updated, changed = await complete_conversation_turn(
        llm,
        convo,
        max_tokens=1024,
        stream_func=_noop_stream,
    )
    assert changed == "changed"
    assert updated[-1].content == LIMITED_ANSWER
    llm._continue_cutoff_response.assert_awaited_once()
    assert llm._continue_cutoff_response.await_args.kwargs["stream_callback"] is None
    assert len(events) == 1 and events[0]["done"] is True
    assert events[0]["content"] == LIMITED_ANSWER


@pytest.mark.asyncio
async def test_complete_turn_preserves_cited_cutoff_draft_when_continuation_is_empty(monkeypatch):
    async def _fake_compact_summary(_llm, _conversation, _max_tokens):
        return "Compressed fallback that should not replace the richer draft."

    monkeypatch.setattr(
        "lib.llm.providers.complete_turn.generate_compact_tool_summary",
        _fake_compact_summary,
    )

    partial = (
        "- The book frames heredity as a live historical controversy rather than a settled legacy [doc:doc-a chunk:7]\n"
        "- Radick shows Mendelian ideas being used in everyday disputes, including arguments about parentage [doc:doc-a chunk:7]\n"
        "- Schoolbook rules about dark and light eyes were treated as decisive evidence in cases that were much messier in practice [doc:doc-a chunk:7]\n"
        "- The available excerpts are still narrow, so broader claims about Mendel's full legacy remain provisional [doc:doc-a chunk:7]"
    )
    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            return_value=LLMResponse(
                text=partial,
                tool_call={},
                stop_reason="max_tokens",
                input_usage=1,
                output_usage=1,
                model="fake",
            )
        ),
        _continue_cutoff_response=AsyncMock(return_value=None),
    )
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Summarize the Gregory Radick evidence with citations."),
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

    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(llm, convo, max_tokens=2048)

    assert changed == "changed"
    assert updated[-1].content == partial
    assert updated[-1]._validated_answer.outcome == "partial"
    assert "Compressed fallback" not in updated[-1].content


@pytest.mark.asyncio
async def test_complete_turn_recovers_from_blank_post_tool_answer_using_list_vector_search_evidence(
    monkeypatch,
):
    monkeypatch.setenv("LLM_POST_TOOL_SYNTHESIS_RETRIES", "2")
    monkeypatch.setenv("RAG_ENFORCE_CHUNK_CITATIONS", "0")
    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            side_effect=[
                LLMResponse(
                    text="",
                    tool_call={},
                    stop_reason="stop",
                    input_usage=1,
                    output_usage=1,
                    model="fake",
                ),
                LLMResponse(
                    text="",
                    tool_call={},
                    stop_reason="stop",
                    input_usage=1,
                    output_usage=1,
                    model="fake",
                ),
                LLMResponse(
                    text=(
                        "- Radick argues that Mendelism's triumph was historically contingent, not inevitable [doc:doc-a chunk:7].\n"
                        "- The excerpts suggest Weldonian alternatives were crowded out rather than simply disproved [doc:doc-a chunk:7]."
                    ),
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
            UserMessage(content="What is the central thesis of this book?"),
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {
                            "chunk_id": 7,
                            "doc_id": "doc-a",
                            "title": "Disputed Inheritance",
                            "text": (
                                "Radick argues that the victory of Mendelian genetics was not the only path biology could "
                                "have taken, and that alternative traditions such as Weldonian biology were historically possible."
                            ),
                        }
                    ]
                },
            ),
        ],
    )

    with selected_publication(convo[-1].result_dict["result"]):
        updated, changed = await complete_conversation_turn(llm, convo, max_tokens=1024)

    assert changed == "changed"
    assert updated[-1].content == (
        "- Radick argues that Mendelism's triumph was historically contingent, not inevitable [doc:doc-a chunk:7].\n"
        "- The excerpts suggest Weldonian alternatives were crowded out rather than simply disproved [doc:doc-a chunk:7]."
    )
    assert updated[-1]._validated_answer.outcome == "partial"
    assert "could not generate a final answer" not in updated[-1].content
    assert llm.get_message.await_count == 3


@pytest.mark.asyncio
async def test_complete_turn_does_not_use_chunk_dump_when_synthesis_fails(monkeypatch):
    monkeypatch.setenv("LLM_ALLOW_EXTRACTIVE_EVIDENCE_UI", "0")
    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            return_value=LLMResponse(
                text="",
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
            UserMessage(content="Explain the paper."),
            ToolMessage(
                content="{}",
                tool_name="whole_document",
                for_whom="assistant",
                arguments={"doc_id": "doc-paper"},
                result_dict={
                    "result": {
                        "text": (
                            "Hallucinations undermine trust when models speak without "
                            "calibrated confidence estimates across domains."
                        ),
                    }
                },
            ),
        ],
    )

    updated, changed = await complete_conversation_turn(llm, convo, max_tokens=1024)

    assert changed == "changed"
    assert updated[-1].content == LIMITED_ANSWER
    assert "Hallucinations undermine trust" not in updated[-1].content
    assert "I found supporting context" not in updated[-1].content


def test_synthesize_cited_extract_still_available_when_extractive_ui_enabled():
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
                            "chunk_id": 11702,
                            "doc_id": "doc-a",
                            "title": "DeepSeek Attention",
                            "text": (
                                "DeepSeek attention compresses key-value cache usage by separating latent "
                                "representations from query heads while preserving enough context for generation."
                            ),
                        }
                    ]
                },
            ),
        ],
    )
    extract = synthesize_cited_extract_from_results(convo)
    assert extract is not None
    assert "DeepSeek attention compresses" in extract


@pytest.mark.asyncio
async def test_complete_turn_recovers_blank_post_tool_whole_document_via_synthesis_retry():
    llm = SimpleNamespace(
        base_args={},
        get_message=AsyncMock(
            side_effect=[
                LLMResponse(
                    text="",
                    tool_call={},
                    stop_reason="stop",
                    input_usage=1,
                    output_usage=1,
                    model="fake",
                ),
                LLMResponse(
                    text=(
                        "This paper defines a calibration loss and reliability diagram [doc:doc-paper chunk:1]. "
                        "Equation (3) scores bin-wise accuracy; Figure 2 shows the curve [doc:doc-paper chunk:1]."
                    ),
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
            UserMessage(
                content="Tell me about this paper and show figures with math explained."
            ),
            ToolMessage(
                content="{}",
                tool_name="whole_document",
                for_whom="assistant",
                arguments={"doc_id": "doc-paper"},
                result_dict={
                    "result": {
                        "type": "document_with_figures",
                        "text": (
                            "The objective combines a cross-entropy term with a calibration penalty. "
                            "Equation (3) defines the reliability score used in Figure 2."
                        ),
                        "figures": [
                            {
                                "type": "image",
                                "title": "Figure 2",
                                "text": "Reliability diagram across ten bins.",
                                "image_url": "/aquillm/document_image/fig-2/",
                            }
                        ],
                    }
                },
            ),
        ],
    )

    with selected_publication([{"doc_id": "doc-paper", "chunk_id": 1, "text": convo[-1].result_dict["result"]["text"]}]):
        updated, changed = await complete_conversation_turn(llm, convo, max_tokens=2048)

    assert changed == "changed"
    assert llm.get_message.await_count == 2
    content = updated[-1].content
    assert "I found supporting context" not in content
    assert content == (
        "This paper defines a calibration loss and reliability diagram [doc:doc-paper chunk:1]. "
        "Equation (3) scores bin-wise accuracy; Figure 2 shows the curve [doc:doc-paper chunk:1]."
    )
    assert updated[-1]._validated_answer.outcome == "partial"
