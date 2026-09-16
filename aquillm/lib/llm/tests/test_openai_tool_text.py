"""Tests for authorization-bound textual tool-call recovery."""

from __future__ import annotations

import pytest

from lib.llm.providers.openai_tool_text import extract_tool_call_from_text


def test_parses_authorized_same_line_tool_call():
    parsed = extract_tool_call_from_text(
        'Tool:vector_search {"search_string":"attensity","top_k":10}',
        [{"name": "vector_search"}],
    )

    assert parsed is not None
    assert parsed["tool_call_name"] == "vector_search"
    assert parsed["tool_call_input"] == {
        "search_string": "attensity",
        "top_k": 10,
    }


def test_parses_authorized_same_line_tool_call_with_empty_arguments():
    parsed = extract_tool_call_from_text(
        "Tool:document_ids {}",
        [{"name": "document_ids"}],
    )

    assert parsed is not None
    assert parsed["tool_call_name"] == "document_ids"
    assert parsed["tool_call_input"] == {}


@pytest.mark.parametrize(
    "text",
    [
        'Tool:unknown {"search_string":"attensity"}',
        'Tool:vector_search {"search_string":',
        'Before searching, Tool:vector_search {"search_string":"attensity"}',
    ],
)
def test_rejects_unauthorized_malformed_or_embedded_same_line_tool_call(text):
    parsed = extract_tool_call_from_text(text, [{"name": "vector_search"}])

    assert parsed is None
