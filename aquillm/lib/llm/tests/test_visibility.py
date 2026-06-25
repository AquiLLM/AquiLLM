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


def test_concise_final_answer_is_displayable():
    assert vis.is_displayable_answer_text("Here is the figure summary.")


def test_short_cited_final_answer_is_displayable():
    assert vis.is_displayable_answer_text("Cited answer [doc:doc-a chunk:7].")


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


def test_visible_stream_forwards_partial_non_interim_answer_before_display_threshold():
    partial = "The paper introduces a calibration objective for confidence scores."
    visible = vis.visible_stream_content(
        partial,
        raw_tools=[{"name": "whole_document"}],
        done=False,
        tool_call_payload=None,
    )
    assert visible == partial


def test_extractive_chunk_dump_is_not_displayable():
    dump = (
        "Here is a concise summary from the retrieved documents:\n"
        "- o at o ( ); o p edic [doc:abc]"
    )
    assert vis.looks_like_extractive_chunk_dump(dump)
    assert not vis.is_displayable_answer_text(dump)


def test_visible_stream_suppresses_tool_code_markup_fragment():
    visible = vis.visible_stream_content(
        "<tool_code> Tool",
        raw_tools=[{"name": "whole_document"}],
        done=False,
        tool_call_payload=None,
    )
    assert visible == ""


def test_sanitize_strips_tool_code_from_persisted_answer():
    text = "Overview text.\n\n<tool_code>{\"name\":\"vector_search\"}</tool_code>\n\nMore detail."
    cleaned = vis.sanitize_assistant_text(text, suppress_interim=False)
    assert "<tool_code>" not in cleaned
    assert "Overview text" in cleaned
    assert "More detail" in cleaned


def test_visible_stream_done_does_not_drop_short_substantive_answer():
    answer = "Figure 2 shows the main result."
    visible = vis.visible_stream_content(
        answer,
        raw_tools=None,
        done=True,
        tool_call_payload=None,
    )
    assert visible == answer
