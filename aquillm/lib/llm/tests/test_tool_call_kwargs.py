"""Tool-call argument normalization before validate_call."""
from __future__ import annotations

from lib.llm.utils.tool_call_kwargs import normalize_tool_call_kwargs


def test_vector_search_maps_query_and_string_top_k():
    out = normalize_tool_call_kwargs(
        "vector_search",
        {"query": "mendel inheritance", "top_k": "8"},
    )
    assert out["search_string"] == "mendel inheritance"
    assert out["top_k"] == 8


def test_vector_search_maps_k_and_float_top_k():
    out = normalize_tool_call_kwargs("vector_search", {"q": "biology", "k": 5.0})
    assert out["search_string"] == "biology"
    assert out["top_k"] == 5


def test_vector_search_keeps_canonical_keys():
    out = normalize_tool_call_kwargs(
        "vector_search",
        {"search_string": "primary", "top_k": 3},
    )
    assert out["search_string"] == "primary"
    assert out["top_k"] == 3


def test_whole_document_untouched():
    raw = {"doc_id": "550e8400-e29b-41d4-a716-446655440000"}
    assert normalize_tool_call_kwargs("whole_document", raw) == raw


def test_search_single_document_coerces_top_k_string():
    out = normalize_tool_call_kwargs(
        "search_single_document",
        {
            "doc_id": "550e8400-e29b-41d4-a716-446655440000",
            "search_string": "test",
            "top_k": "4",
        },
    )
    assert out["top_k"] == 4
