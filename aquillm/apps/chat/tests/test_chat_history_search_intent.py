"""Tests for past-chat (conversation history) search intent gating."""
from __future__ import annotations

from aquillm.llm import ToolChoice
from apps.chat.consumers.chat_receive import (
    _configure_append_tools,
    _looks_like_chat_history_search_request,
)

_MEMORY_TOOL = object()
_DOC_TOOL = object()


def test_recall_phrasings_match():
    positives = [
        "What did we discuss about the calibration earlier?",
        "Remind me what we decided in a previous conversation.",
        "Search my past chats for the telescope alignment notes.",
        "Look through our earlier conversations about the grant.",
        "Check my chat history for the deadline we agreed on.",
        "What did we talk about last time?",
    ]
    for text in positives:
        assert _looks_like_chat_history_search_request(text), text


def test_ordinary_and_document_phrasings_do_not_match():
    negatives = [
        "Can you explain how a rotating black hole forms?",
        "Please search the selected documents for calibration notes.",
        "Show me the figures from the paper.",
        "What is apple pie made of?",
    ]
    for text in negatives:
        assert not _looks_like_chat_history_search_request(text), text


def test_history_request_forces_memory_tool():
    tools, tool_choice = _configure_append_tools(
        message_content="What did we discuss in an earlier conversation about quasars?",
        all_tools=[_DOC_TOOL, _MEMORY_TOOL],
        document_tools=[_DOC_TOOL],
        memory_tools=[_MEMORY_TOOL],
    )
    assert tools == [_MEMORY_TOOL]
    assert tool_choice == ToolChoice(type="any")


def test_document_request_still_wins_over_memory_when_no_history_cue():
    tools, tool_choice = _configure_append_tools(
        message_content="Please search the selected documents for the calibration notes.",
        all_tools=[_DOC_TOOL, _MEMORY_TOOL],
        document_tools=[_DOC_TOOL],
        memory_tools=[_MEMORY_TOOL],
    )
    assert tools == [_DOC_TOOL]
    assert tool_choice == ToolChoice(type="any")
