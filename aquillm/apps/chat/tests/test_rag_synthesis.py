"""Tests for evidence-first synthesis (Task 5)."""
from __future__ import annotations

import pytest

from lib.llm.providers import visibility
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_synthesis import synthesize_from_evidence


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeCompleteLLM:
    """Stub LLM whose ``complete`` appends a fixed assistant reply."""

    def __init__(self, reply: str):
        self.reply = reply
        self.complete_calls = 0

    async def complete(self, convo, max_tokens, stream_func=None):
        self.complete_calls += 1
        return (
            convo + [AssistantMessage(content=self.reply, stop_reason="end_turn")],
            "changed",
        )


class _NoCallLLM:
    def __init__(self):
        self.complete_calls = 0

    async def complete(self, *args, **kwargs):
        self.complete_calls += 1
        raise AssertionError("complete() must not run when there are no results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_results(*, image: bool = False) -> dict:
    chunk = {
        "rank": 1,
        "chunk_id": 1,
        "doc_id": "doc-a",
        "title": "Paper A",
        "text": "Calibration uses flat fields and dark frames to remove signatures.",
        "citation": "[doc:doc-a chunk:1]",
    }
    if image:
        chunk["image_url"] = "/aquillm/document_image/doc-a/"
    return {
        "result": [chunk],
        "retrieval_status": "results_found",
        "retrieved_count": 1,
        "retrieved_documents": ["Paper A"],
    }


def _raw_no_results(query: str = "dark matter") -> dict:
    return {
        "result": [],
        "retrieval_status": "no_results",
        "retrieval_message": (
            f'I searched the selected documents for "{query}", '
            "but retrieval returned no relevant passages."
        ),
    }


def _post_tool_convo(user_text: str, raw_result: dict) -> Conversation:
    arguments = {"search_string": "calibration", "top_k": 5}
    return Conversation(
        system="sys",
        messages=[
            UserMessage(content=user_text),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="t1",
                tool_call_name="vector_search",
                tool_call_input=arguments,
            ),
            ToolMessage(
                tool_name="vector_search",
                for_whom="assistant",
                content="{}",
                arguments=arguments,
                result_dict=raw_result,
            ),
        ],
    )


def _packet(raw_result: dict, query: str = "calibration"):
    return build_evidence_packet(
        raw_result, query=query, search_scope="selected documents"
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_uses_llm_answer_when_usable():
    raw = _raw_results()
    convo = _post_tool_convo("summarize the calibration method", raw)
    reply = (
        "The paper calibrates by subtracting dark frames and dividing by flat "
        "fields to remove instrument signatures [doc:doc-a chunk:1]."
    )
    llm = _FakeCompleteLLM(reply)

    result = await synthesize_from_evidence(llm, convo, _packet(raw))

    assert llm.complete_calls == 1
    assert result[-1].content == reply


# ---------------------------------------------------------------------------
# Extractive fallback (always on for direct RAG)
# ---------------------------------------------------------------------------

async def test_blank_synthesis_uses_extractive_fallback(monkeypatch):
    monkeypatch.delenv("LLM_ALLOW_EXTRACTIVE_EVIDENCE_UI", raising=False)
    raw = _raw_results()
    convo = _post_tool_convo("summarize the calibration method", raw)
    llm = _FakeCompleteLLM("")

    result = await synthesize_from_evidence(llm, convo, _packet(raw))

    content = result[-1].content
    assert content.strip()
    assert "[doc:doc-a chunk:1]" in content
    assert "flat fields" in content


async def test_failure_text_replaced_by_extractive():
    raw = _raw_results()
    convo = _post_tool_convo("summarize the calibration method", raw)
    failure = visibility.clean_response_failure_text(after_tool_result=True)
    llm = _FakeCompleteLLM(failure)

    result = await synthesize_from_evidence(llm, convo, _packet(raw))

    content = result[-1].content
    assert content != failure
    assert "[doc:doc-a chunk:1]" in content


# ---------------------------------------------------------------------------
# No results -> transparent notice, no LLM call
# ---------------------------------------------------------------------------

async def test_no_results_returns_notice_without_llm():
    raw = _raw_no_results("gravitational waves")
    convo = _post_tool_convo("search for gravitational waves", raw)
    llm = _NoCallLLM()

    result = await synthesize_from_evidence(
        llm, convo, _packet(raw, query="gravitational waves")
    )

    assert llm.complete_calls == 0
    content = result[-1].content
    assert isinstance(result[-1], AssistantMessage)
    assert "no relevant passages" in content.lower()


# ---------------------------------------------------------------------------
# Figure requests ensure markdown images
# ---------------------------------------------------------------------------

async def test_figure_request_appends_markdown_image():
    raw = _raw_results(image=True)
    convo = _post_tool_convo("show me the figure for calibration", raw)
    reply = "The figure shows calibration drift across magnitude bins [doc:doc-a chunk:1]."
    llm = _FakeCompleteLLM(reply)

    result = await synthesize_from_evidence(llm, convo, _packet(raw))

    content = result[-1].content
    assert "/aquillm/document_image/doc-a/" in content
    assert "![" in content


async def test_no_figure_appended_when_not_requested():
    raw = _raw_results(image=True)
    convo = _post_tool_convo("explain the calibration method in detail", raw)
    reply = "Calibration subtracts dark frames and divides by flat fields [doc:doc-a chunk:1]."
    llm = _FakeCompleteLLM(reply)

    result = await synthesize_from_evidence(llm, convo, _packet(raw))

    assert result[-1].content == reply


async def test_figure_not_duplicated_when_already_present():
    raw = _raw_results(image=True)
    convo = _post_tool_convo("show me the figure", raw)
    reply = (
        "Here is the figure [doc:doc-a chunk:1]:\n\n"
        "![existing](/aquillm/document_image/doc-a/)"
    )
    llm = _FakeCompleteLLM(reply)

    result = await synthesize_from_evidence(llm, convo, _packet(raw))

    content = result[-1].content
    assert content.count("/aquillm/document_image/doc-a/") == 1
