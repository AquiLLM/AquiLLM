"""Single-document tool compatibility with graph-expanded retrieval."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from apps.chat.refs import CollectionsRef
from apps.chat.services.tool_wiring import documents as document_tools

_DOC_ID = UUID("00000000-0000-0000-0000-000000000017")


def test_single_document_search_keeps_scope_payload_and_real_citation(monkeypatch):
    collection = SimpleNamespace(user_can_view=lambda _user: True)
    document = SimpleNamespace(
        id=_DOC_ID,
        title="Only authorized document",
        collection=collection,
        image_file=None,
    )
    chunk = SimpleNamespace(
        id=71,
        doc_id=_DOC_ID,
        chunk_number=3,
        modality="text",
        content="Context reached through the graph.",
        graph_score=0.88,
        graph_path=("private", "node"),
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        document_tools,
        "_resolve_doc_uuid",
        lambda *_args: (_DOC_ID, ""),
    )
    monkeypatch.setattr(
        document_tools.Document,
        "get_by_id",
        lambda _document_id: document,
    )

    def fake_search(query, top_k, docs):
        seen["query"] = query
        seen["top_k"] = top_k
        seen["docs"] = docs
        return (
            (),
            (),
            (chunk,),
            {
                "doc_count": 1,
                "graph_status": "hit",
                "graph_seed_count": 1,
                "graph_candidate_count": 1,
            },
        )

    monkeypatch.setattr(
        document_tools.TextChunk,
        "text_chunk_search",
        fake_search,
    )
    tool = document_tools.search_single_document_tool(
        user=object(),
        col_ref=CollectionsRef([3]),
    )

    result = tool(
        doc_id=str(_DOC_ID),
        search_string="related context",
        top_k=4,
    )

    assert seen == {
        "query": "related context",
        "top_k": 4,
        "docs": [document],
    }
    assert result["result"] == [
        {
            "rank": 1,
            "chunk_id": 71,
            "doc_id": str(_DOC_ID),
            "chunk": 3,
            "title": "Only authorized document",
            "citation": f"[doc:{_DOC_ID} chunk:71]",
            "text": "Context reached through the graph.",
        }
    ]
    assert "retrieval_diagnostics" not in result
    assert not any(key.startswith("graph_") for key in result)


def test_single_document_no_result_diagnostics_strip_graph_fields(monkeypatch):
    collection = SimpleNamespace(user_can_view=lambda _user: True)
    document = SimpleNamespace(
        id=_DOC_ID,
        title="Only authorized document",
        collection=collection,
        image_file=None,
    )
    monkeypatch.setattr(
        document_tools,
        "_resolve_doc_uuid",
        lambda *_args: (_DOC_ID, ""),
    )
    monkeypatch.setattr(
        document_tools.Document,
        "get_by_id",
        lambda _document_id: document,
    )
    monkeypatch.setattr(
        document_tools.TextChunk,
        "text_chunk_search",
        lambda *_args: (
            (),
            (),
            (),
            {
                "doc_count": 1,
                "vector_error": None,
                "graph_status": "miss",
                "graph_seed_count": 0,
            },
        ),
    )
    tool = document_tools.search_single_document_tool(
        user=object(),
        col_ref=CollectionsRef([3]),
    )

    result = tool(
        doc_id=str(_DOC_ID),
        search_string="missing context",
        top_k=4,
    )

    assert result["retrieval_diagnostics"] == {
        "doc_count": 1,
        "vector_error": None,
    }
