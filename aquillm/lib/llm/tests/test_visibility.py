"""Tests for assistant presentation policy (display vs persistence)."""
from __future__ import annotations

from aquillm.llm import AssistantMessage
from lib.llm.providers import visibility as vis


def test_status_only_retrieving_paper_is_not_displayable():
    assert vis.looks_like_status_only("Retrieving the paper...")
    assert not vis.is_displayable_answer_text("Retrieving the paper...")


def test_substantial_answer_with_citation_is_displayable():
    text = (
        "The paper defines calibration as aligning confidence with accuracy [doc:abc chunk:1]. "
        "It then compares several estimators across domains."
    )
    assert vis.is_displayable_answer_text(text)


def test_assistant_tool_call_row_has_empty_frontend_content():
    msg = AssistantMessage(
        content="I'll retrieve the passage now.",
        stop_reason="tool_use",
        tool_call_id="tc-1",
        tool_call_name="vector_search",
        tool_call_input={"search_string": "memory"},
    )
    assert vis.assistant_content_for_frontend(msg) == ""


def test_should_not_append_sources_for_status_stub():
    assert not vis.should_append_citation_sources("Retrieving the paper...")


def test_visible_stream_done_suppresses_status_stub():
    visible = vis.visible_stream_content(
        "Retrieving the paper...",
        raw_tools=[{"name": "vector_search"}],
        done=True,
        tool_call_payload=None,
    )
    assert visible == ""
