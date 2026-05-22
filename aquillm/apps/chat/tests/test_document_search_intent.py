"""Regression tests for explicit document-search requests in chat."""

from __future__ import annotations

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
