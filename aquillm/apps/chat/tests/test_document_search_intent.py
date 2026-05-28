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


def test_regular_chat_keeps_all_tools_on_auto():
    message = "Can you explain why the sky subtraction step matters?"

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
