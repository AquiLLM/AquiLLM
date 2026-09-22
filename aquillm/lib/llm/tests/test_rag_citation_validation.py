"""Tests for strict, verifiable RAG citation handling."""
from __future__ import annotations

from lib.llm.providers.rag_citations import (
    collect_allowed_chunk_citations,
    find_invalid_citations,
    response_has_required_citations,
    synthesize_cited_extract_from_results,
    synthesize_doc_level_extract_from_results,
)
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage


def test_collect_allowed_chunk_citations_from_verbose_and_compact_rows():
    convo = Conversation(
        system="sys",
        messages=[
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {"chunk_id": 7, "doc_id": "doc-a", "title": "Doc A", "text": "alpha"},
                        {"i": 9, "d": "doc-b", "n": "Doc B", "x": "beta"},
                    ]
                },
            )
        ],
    )
    allowed = collect_allowed_chunk_citations(convo)
    assert "[doc:doc-a chunk:7]" in allowed
    assert "[doc:doc-b chunk:9]" in allowed


def test_graph_metadata_cannot_create_pseudo_citations():
    convo = Conversation(
        system="sys",
        messages=[
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {
                            "chunk_id": 17,
                            "doc_id": "doc-real",
                            "citation": "[doc:doc-real chunk:17]",
                            "text": "real graph-expanded chunk",
                            "graph_path": "[doc:private chunk:999]",
                            "graph_entity": "private canonical node",
                            "graph_score": 0.9,
                        }
                    ],
                    "retrieval_diagnostics": {
                        "graph_status": "hit",
                        "graph_version_signature": "a" * 64,
                    },
                },
            )
        ],
    )

    assert collect_allowed_chunk_citations(convo) == {
        "[doc:doc-real chunk:17]"
    }


def test_synthesize_doc_level_extract_from_whole_document_payload():
    convo = Conversation(
        system="sys",
        messages=[
            ToolMessage(
                content="{}",
                tool_name="whole_document",
                for_whom="assistant",
                arguments={"doc_id": "doc-paper"},
                result_dict={
                    "result": {
                        "type": "document_with_figures",
                        "text": (
                            "The model minimizes a calibration loss L(theta). "
                            "Figure 2 plots the reliability curve across bins."
                        ),
                        "figures": [],
                    }
                },
            )
        ],
    )
    extract = synthesize_doc_level_extract_from_results(convo)
    assert extract is not None
    assert "calibration loss" in extract
    assert "[doc:doc-paper]" in extract


def test_collect_allowed_chunk_citations_accepts_explicit_ref_fields():
    convo = Conversation(
        system="sys",
        messages=[
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {"citation": "[doc:doc-c chunk:11]", "text": "gamma"},
                        {"ref": "prefix [doc:doc-d chunk:12] suffix", "x": "delta"},
                    ]
                },
            )
        ],
    )
    allowed = collect_allowed_chunk_citations(convo)
    assert "[doc:doc-c chunk:11]" in allowed
    assert "[doc:doc-d chunk:12]" in allowed


def test_collect_allowed_chunk_citations_accepts_whole_document_metadata():
    convo = Conversation(
        system="sys",
        messages=[
            ToolMessage(
                content="{}",
                tool_name="whole_document",
                for_whom="assistant",
                result_dict={
                    "result": "[doc:doc-paper chunk:21]\nA cited passage.",
                    "citation_chunks": [
                        {
                            "doc_id": "doc-paper",
                            "chunk_id": 21,
                            "citation": "[doc:doc-paper chunk:21]",
                        }
                    ],
                },
            )
        ],
    )

    assert collect_allowed_chunk_citations(convo) == {
        "[doc:doc-paper chunk:21]"
    }


def test_whole_document_citation_metadata_is_not_truncated_by_search_row_cap():
    citation_chunks = [
        {
            "doc_id": "doc-paper",
            "chunk_id": chunk_id,
            "citation": f"[doc:doc-paper chunk:{chunk_id}]",
        }
        for chunk_id in range(1, 46)
    ]
    convo = Conversation(
        system="sys",
        messages=[
            ToolMessage(
                content="{}",
                tool_name="whole_document",
                for_whom="assistant",
                result_dict={
                    "result": "whole document",
                    "citation_chunks": citation_chunks,
                },
            )
        ],
    )

    allowed = collect_allowed_chunk_citations(convo, max_rows_per_message=1)

    assert len(allowed) == 45
    assert "[doc:doc-paper chunk:45]" in allowed


def test_response_has_required_citations_and_rejects_unknown_refs():
    allowed = {"[doc:doc-a chunk:7]"}
    assert response_has_required_citations("Fact from source [doc:doc-a chunk:7].", allowed)
    assert not response_has_required_citations("Fact with no cite.", allowed)
    invalid = find_invalid_citations(
        "Wrong cite [doc:doc-a chunk:999] plus right [doc:doc-a chunk:7].",
        allowed,
    )
    assert invalid == ["[doc:doc-a chunk:999]"]


def test_response_has_required_citations_rejects_uncited_bullets():
    allowed = {"[doc:doc-a chunk:7]"}
    answer = "- Supported claim [doc:doc-a chunk:7]\n- Unsupported claim without cite"
    assert not response_has_required_citations(answer, allowed)


def test_response_has_required_citations_allows_uncited_connective_sentence():
    allowed = {"[doc:doc-a chunk:7]"}
    answer = (
        "- Core finding from the paper [doc:doc-a chunk:7]\n"
        "Overall, this suggests the pattern is robust across experiments."
    )
    assert response_has_required_citations(answer, allowed)


def test_response_has_required_citations_allows_indented_sub_bullets_under_cited_heading():
    allowed = {"[doc:doc-a chunk:7]"}
    answer = (
        "Figure 27 - Degrees of wrinkledness [doc:doc-a chunk:7]\n"
        "    - Telegraph seeds 1-4\n"
        "    - Telephone seeds 5-6\n"
        "    - Lightning seeds 13-15"
    )
    assert response_has_required_citations(answer, allowed)


def test_synthesize_cited_extract_from_results_includes_refs():
    convo = Conversation(
        system="sys",
        messages=[
            ToolMessage(
                content="{}",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {
                            "chunk_id": 7,
                            "doc_id": "doc-a",
                            "title": "Doc A",
                            "text": "Alpha finding with supporting detail.",
                        }
                    ]
                },
            )
        ],
    )
    fallback = synthesize_cited_extract_from_results(convo)
    assert fallback is not None
    assert "Alpha finding" in fallback
    assert "[doc:doc-a chunk:7]" in fallback
