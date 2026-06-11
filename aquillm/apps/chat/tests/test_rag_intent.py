"""Tests for RAG intent classification (Task 1)."""
from __future__ import annotations

import pytest

from apps.chat.services.rag_intent import ChatIntent, classify_chat_message


# ---------------------------------------------------------------------------
# Explicit-search detection
# ---------------------------------------------------------------------------

def test_explicit_document_search_requires_rag():
    result = classify_chat_message(
        "Please search the selected documents for calibration notes.",
        selected_collection_ids=[1],
    )
    assert result.requires_rag is True
    assert result.reason == "explicit_search"


def test_explicit_search_without_collections_still_requires_rag():
    """Explicit search verb + document noun → RAG regardless of collection list."""
    result = classify_chat_message(
        "Search the docs for dark matter.",
        selected_collection_ids=[],
    )
    assert result.requires_rag is True


# ---------------------------------------------------------------------------
# Figure-request detection
# ---------------------------------------------------------------------------

def test_figure_request_requires_rag():
    result = classify_chat_message(
        "Can you show me some of the figures from it with context please",
        selected_collection_ids=[],
    )
    assert result.requires_rag is True
    assert result.wants_figures is True
    assert result.reason == "figure_request"


def test_followup_figure_request_no_source_word():
    result = classify_chat_message(
        "ok can you show me the figures though?",
        selected_collection_ids=[],
    )
    assert result.requires_rag is True
    assert result.wants_figures is True


# ---------------------------------------------------------------------------
# Collection-backed document question (new expansion rule)
# ---------------------------------------------------------------------------

def test_collection_backed_question_requires_rag():
    result = classify_chat_message(
        "What does this paper say about X?",
        selected_collection_ids=[42],
    )
    assert result.requires_rag is True
    assert result.reason == "collection_backed_question"


def test_collection_backed_how_question():
    result = classify_chat_message(
        "How does this document describe the calibration process?",
        selected_collection_ids=[1, 2],
    )
    assert result.requires_rag is True
    assert result.reason == "collection_backed_question"


def test_collection_backed_explain_question():
    result = classify_chat_message(
        "Explain the findings from this article.",
        selected_collection_ids=[7],
    )
    assert result.requires_rag is True
    assert result.reason == "collection_backed_question"


def test_collection_backed_question_no_collections_returns_false():
    """Same text but no collections selected → no RAG."""
    result = classify_chat_message(
        "What does this paper say about X?",
        selected_collection_ids=[],
    )
    assert result.requires_rag is False


# ---------------------------------------------------------------------------
# Brand-new chat / no retrieval needed
# ---------------------------------------------------------------------------

def test_brand_new_chat_no_rag():
    result = classify_chat_message(
        "brand new chat",
        selected_collection_ids=[],
    )
    assert result.requires_rag is False
    assert result.requires_local_tools is False
    assert result.is_retry is False
    assert result.reason == "no_retrieval_needed"


def test_regular_followup_no_rag():
    result = classify_chat_message(
        "Can you show how they predict a rotating black hole?",
        selected_collection_ids=[],
    )
    assert result.requires_rag is False


# ---------------------------------------------------------------------------
# Local-tool detection
# ---------------------------------------------------------------------------

def test_local_tool_request_sets_requires_local_tools():
    result = classify_chat_message(
        "Please subtract the sky from object file 1 using sky file 2.",
        selected_collection_ids=[],
    )
    assert result.requires_rag is False
    assert result.requires_local_tools is True
    assert result.reason == "local_tool_request"


def test_fits_processing_requires_local_tools():
    result = classify_chat_message(
        "Can you process these FITS files?",
        selected_collection_ids=[],
    )
    assert result.requires_local_tools is True
    assert result.requires_rag is False


# ---------------------------------------------------------------------------
# Retry detection
# ---------------------------------------------------------------------------

def test_retry_request_is_retry():
    result = classify_chat_message(
        "try again",
        selected_collection_ids=[],
        prior_tools=["some_tool"],
        prior_tool_choice=None,
    )
    assert result.is_retry is True
    assert result.reason == "retry_request"


def test_retry_with_no_prior_tools_not_rag():
    result = classify_chat_message(
        "try again",
        selected_collection_ids=[],
        prior_tools=None,
        prior_tool_choice=None,
    )
    assert result.is_retry is True
    assert result.requires_rag is False


def test_retry_with_prior_tools_requires_rag():
    result = classify_chat_message(
        "retry",
        selected_collection_ids=[],
        prior_tools=["doc_tool"],
        prior_tool_choice=None,
    )
    assert result.is_retry is True
    assert result.requires_rag is True


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_chat_intent_dataclass():
    result = classify_chat_message("hello", selected_collection_ids=[])
    assert isinstance(result, ChatIntent)
    assert hasattr(result, "requires_rag")
    assert hasattr(result, "wants_figures")
    assert hasattr(result, "wants_whole_document")
    assert hasattr(result, "is_retry")
    assert hasattr(result, "requires_local_tools")
    assert hasattr(result, "reason")
