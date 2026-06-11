"""Tests for RAG retrieval query builder (Task 2)."""
from __future__ import annotations

import pytest

from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.conversation import Conversation

from apps.chat.services.rag_query import build_retrieval_query


def _make_convo(*messages) -> Conversation:
    return Conversation(system="test", messages=list(messages))


def _search_turn(search_string: str, doc_titles: list[str] | None = None) -> list:
    """Return [UserMessage, AssistantMessage(tool_call), ToolMessage(result)] for a search."""
    result_dict = {
        "retrieval_status": "results_found",
        "retrieved_count": 1,
    }
    if doc_titles:
        result_dict["retrieved_documents"] = doc_titles
    return [
        UserMessage(content=f"search for {search_string}"),
        AssistantMessage(
            stop_reason="tool_use",
            tool_call_id="call_1",
            tool_call_name="vector_search",
            tool_call_input={"search_string": search_string, "top_k": 10},
            content="",
        ),
        ToolMessage(
            tool_name="vector_search",
            for_whom="assistant",
            content="{}",
            result_dict=result_dict,
        ),
        AssistantMessage(
            stop_reason="end_turn",
            content=f"Here are results about {search_string}.",
        ),
    ]


# ---------------------------------------------------------------------------
# Plain query passthrough
# ---------------------------------------------------------------------------

def test_plain_query_returned_as_is():
    convo = _make_convo()
    result = build_retrieval_query(convo, "What is dark matter?")
    assert result == "What is dark matter?"


def test_whitespace_trimmed():
    convo = _make_convo()
    result = build_retrieval_query(convo, "  dark energy  ")
    assert result == "dark energy"


# ---------------------------------------------------------------------------
# Retry: reuse last vector_search query
# ---------------------------------------------------------------------------

def test_retry_reuses_last_vector_search_query():
    msgs = _search_turn("dark matter")
    convo = _make_convo(*msgs)
    result = build_retrieval_query(convo, "try again")
    assert result == "dark matter"


def test_retry_reuses_most_recent_query_when_multiple_turns():
    msgs1 = _search_turn("gravitational waves")
    msgs2 = _search_turn("black hole merger")
    convo = _make_convo(*msgs1, *msgs2)
    result = build_retrieval_query(convo, "retry")
    assert result == "black hole merger"


def test_retry_with_no_prior_search_falls_back_to_text():
    convo = _make_convo()
    result = build_retrieval_query(convo, "try again")
    assert result == "try again"


def test_retry_case_insensitive():
    msgs = _search_turn("neutron star")
    convo = _make_convo(*msgs)
    result = build_retrieval_query(convo, "Try Again")
    assert result == "neutron star"


# ---------------------------------------------------------------------------
# Pronoun follow-ups: prepend document title
# ---------------------------------------------------------------------------

def test_pronoun_it_prepends_recent_document_title():
    msgs = _search_turn("dark matter", doc_titles=["Doc A"])
    convo = _make_convo(*msgs)
    result = build_retrieval_query(convo, "explain the math in it")
    assert "Doc A" in result
    assert "explain the math in it" in result


def test_pronoun_this_prepends_recent_document_title():
    msgs = _search_turn("calibration", doc_titles=["Calibration Paper"])
    convo = _make_convo(*msgs)
    result = build_retrieval_query(convo, "summarize this for me")
    assert "Calibration Paper" in result


def test_pronoun_they_prepends_recent_document_title():
    msgs = _search_turn("quasars", doc_titles=["Quasar Survey"])
    convo = _make_convo(*msgs)
    result = build_retrieval_query(convo, "how do they form?")
    assert "Quasar Survey" in result


def test_pronoun_without_prior_retrieval_returns_text_as_is():
    convo = _make_convo()
    result = build_retrieval_query(convo, "explain it")
    assert result == "explain it"


def test_no_pronoun_in_followup_returns_text_as_is():
    msgs = _search_turn("dark matter", doc_titles=["Doc A"])
    convo = _make_convo(*msgs)
    result = build_retrieval_query(convo, "What is the Hubble constant?")
    assert result == "What is the Hubble constant?"


# ---------------------------------------------------------------------------
# LLM rewrite (mocked, gated behind RAG_QUERY_REWRITE_ENABLED)
# ---------------------------------------------------------------------------

def test_rewrite_disabled_returns_plain_query(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_REWRITE_ENABLED", "0")
    convo = _make_convo()
    result = build_retrieval_query(convo, "describe the calibration method")
    assert result == "describe the calibration method"


def test_rewrite_enabled_calls_rewrite_function(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_REWRITE_ENABLED", "1")
    rewrite_called_with = []

    def _fake_rewrite(text: str, conversation) -> str:
        rewrite_called_with.append(text)
        return f"REWRITTEN: {text}"

    from apps.chat.services import rag_query
    monkeypatch.setattr(rag_query, "_llm_rewrite_query", _fake_rewrite)

    convo = _make_convo()
    result = build_retrieval_query(convo, "describe the calibration method")
    assert result == "REWRITTEN: describe the calibration method"
    assert rewrite_called_with == ["describe the calibration method"]
