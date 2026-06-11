"""Regression tests for explicit document-search requests in chat."""

from __future__ import annotations

import pytest

from aquillm.llm import ToolChoice
from apps.chat.consumers.chat_receive import _configure_append_tools
from apps.chat.refs import CollectionsRef
from apps.chat.tests.chat_message_test_support import _test_document_ids, _test_image_result_tool
from apps.chat.services.tool_wiring.documents import vector_search_tool


def test_explicit_document_search_request_requires_document_tool_call():
    message = "Please search the selected documents for the instrument calibration notes."

    tools, tool_choice = _configure_append_tools(
        message_content=message,
        all_tools=[_test_document_ids, _test_image_result_tool],
        document_tools=[_test_document_ids],
    )

    assert tools == [_test_document_ids]
    assert tool_choice == ToolChoice(type="any")


def test_figure_request_from_prior_document_requires_document_tool_call():
    message = "Can you show me some of the figures from it with context please"

    tools, tool_choice = _configure_append_tools(
        message_content=message,
        all_tools=[_test_document_ids, _test_image_result_tool],
        document_tools=[_test_document_ids],
    )

    assert tools == [_test_document_ids]
    assert tool_choice == ToolChoice(type="any")


def test_followup_figure_request_without_source_word_requires_document_tool_call():
    message = "ok can you show me the figures though?"

    tools, tool_choice = _configure_append_tools(
        message_content=message,
        all_tools=[_test_document_ids, _test_image_result_tool],
        document_tools=[_test_document_ids],
    )

    assert tools == [_test_document_ids]
    assert tool_choice == ToolChoice(type="any")


def test_retry_request_reuses_prior_tool_intent():
    message = "try again"

    tools, tool_choice = _configure_append_tools(
        message_content=message,
        all_tools=[_test_document_ids, _test_image_result_tool],
        document_tools=[_test_document_ids],
        prior_user_tools=[_test_document_ids],
        prior_user_tool_choice=ToolChoice(type="any"),
    )

    assert tools == [_test_document_ids]
    assert tool_choice == ToolChoice(type="any")


def test_regular_followup_omits_tools_for_direct_answer():
    message = "Can you show how they predict a rotating black hole?"

    tools, tool_choice = _configure_append_tools(
        message_content=message,
        all_tools=[_test_document_ids, _test_image_result_tool],
        document_tools=[_test_document_ids],
    )

    assert tools == []
    assert tool_choice is None


def test_astronomy_processing_request_keeps_all_tools_on_auto():
    message = "Please subtract the sky from object file 1 using sky file 2."

    tools, tool_choice = _configure_append_tools(
        message_content=message,
        all_tools=[_test_document_ids, _test_image_result_tool],
        document_tools=[_test_document_ids],
    )

    assert tools == [_test_document_ids, _test_image_result_tool]
    assert tool_choice == ToolChoice(type="auto")


def test_vector_search_prompt_requires_search_scope_and_sources_in_final_answer():
    tool = vector_search_tool(user=object(), col_ref=CollectionsRef([]))
    description = tool.llm_definition["description"]

    assert "After using this tool" in description
    assert "searched the selected documents" in description
    assert "cite or name the documents" in description


# ---------------------------------------------------------------------------
# RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED feature (Task 6)
# ---------------------------------------------------------------------------

def test_collection_backed_question_attaches_tools_when_flag_on(monkeypatch):
    """Collection-backed question + non-empty collections → document_tools with 'any'."""
    monkeypatch.setenv("RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED", "1")
    message = "What does this paper say about spectral calibration?"

    tools, tool_choice = _configure_append_tools(
        message_content=message,
        all_tools=[_test_document_ids, _test_image_result_tool],
        document_tools=[_test_document_ids],
        selected_collection_ids=[42],
    )

    assert tools == [_test_document_ids]
    assert tool_choice == ToolChoice(type="any")


def test_collection_backed_question_no_tools_without_collections(monkeypatch):
    """Same message but no collections selected → no auto-attach."""
    monkeypatch.setenv("RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED", "1")
    message = "What does this paper say about spectral calibration?"

    tools, tool_choice = _configure_append_tools(
        message_content=message,
        all_tools=[_test_document_ids, _test_image_result_tool],
        document_tools=[_test_document_ids],
        selected_collection_ids=[],
    )

    assert tools == []
    assert tool_choice is None


def test_collection_backed_no_tools_when_flag_off(monkeypatch):
    """Flag off → collection-backed path is skipped."""
    monkeypatch.setenv("RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED", "0")
    message = "What does this paper say about spectral calibration?"

    tools, tool_choice = _configure_append_tools(
        message_content=message,
        all_tools=[_test_document_ids, _test_image_result_tool],
        document_tools=[_test_document_ids],
        selected_collection_ids=[42],
    )

    assert tools == []
    assert tool_choice is None


def test_explicit_search_still_attaches_tools_regardless_of_collection_ids(monkeypatch):
    """Explicit search triggers the old path regardless of collection_ids flag."""
    monkeypatch.setenv("RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED", "0")
    message = "Please search the selected documents for calibration notes."

    tools, tool_choice = _configure_append_tools(
        message_content=message,
        all_tools=[_test_document_ids, _test_image_result_tool],
        document_tools=[_test_document_ids],
        selected_collection_ids=[],
    )

    assert tools == [_test_document_ids]
    assert tool_choice == ToolChoice(type="any")
