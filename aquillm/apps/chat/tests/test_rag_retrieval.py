"""Tests for deterministic multi-query RAG result fusion."""
from __future__ import annotations

from importlib.util import find_spec

from lib.llm.providers.image_context import serialize_tool_result_for_llm


def _row(rank: int, chunk_id: int, doc_id: str) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "title": f"Paper {doc_id}",
        "text": f"Evidence from {doc_id}:{chunk_id}",
        "citation": f"[doc:{doc_id} chunk:{chunk_id}]",
    }


def _payload(*rows: dict, diagnostics: dict | None = None) -> dict:
    payload = {
        "result": list(rows),
        "retrieval_status": "results_found",
        "retrieved_count": len(rows),
        "retrieved_documents": sorted({row["title"] for row in rows}),
    }
    if diagnostics is not None:
        payload["_retrieval_diagnostics"] = diagnostics
    return payload


def _merge(results: list[dict], limit: int) -> dict:
    spec = find_spec("apps.chat.services.rag_retrieval")
    assert spec is not None, "rag_retrieval must provide rank fusion"
    from apps.chat.services.rag_retrieval import merge_ranked_tool_results

    return merge_ranked_tool_results(results, limit=limit)


def test_merge_deduplicates_citations_and_preserves_cross_query_evidence():
    shared = _row(2, 2, "shared")
    merged = _merge(
        [
            _payload(_row(1, 1, "a"), shared),
            _payload({**shared, "rank": 1}, _row(2, 3, "b")),
        ],
        limit=3,
    )

    assert [row["citation"] for row in merged["result"]] == [
        "[doc:shared chunk:2]",
        "[doc:a chunk:1]",
        "[doc:b chunk:3]",
    ]
    assert merged["retrieved_count"] == 3
    assert merged["retrieved_documents"] == ["Paper a", "Paper b", "Paper shared"]


def test_merge_private_graph_diagnostics_stay_out_of_llm_text():
    merged = _merge(
        [
            _payload(
                _row(1, 1, "a"),
                diagnostics={
                    "graph_status": "hit",
                    "graph_candidate_count": 1,
                    "graph_path": ["private-node"],
                },
            )
        ],
        limit=3,
    )

    assert merged["_retrieval_diagnostics"]["graph_status"] == "hit"
    assert "graph_status" not in serialize_tool_result_for_llm(merged)
    assert "private-node" not in serialize_tool_result_for_llm(merged)


def test_merge_aggregates_safe_graph_diagnostics_with_hit_precedence():
    merged = _merge(
        [
            _payload(
                _row(1, 1, "a"),
                diagnostics={
                    "graph_status": "miss",
                    "graph_ms": 1.25,
                    "graph_seed_count": 2,
                    "graph_candidate_count": 0,
                    "graph_path": ["private-a"],
                },
            ),
            _payload(
                _row(1, 2, "b"),
                diagnostics={
                    "graph_status": "hit",
                    "graph_ms": 2.5,
                    "graph_seed_count": 3,
                    "graph_candidate_count": 1,
                    "graph_path": ["private-b"],
                },
            ),
        ],
        limit=3,
    )

    diagnostics = merged["_retrieval_diagnostics"]
    assert diagnostics == {
        "graph_status": "hit",
        "graph_ms": 3.75,
        "graph_seed_count": 5,
        "graph_candidate_count": 1,
    }
    assert "private-a" not in repr(diagnostics)
    assert "private-b" not in repr(diagnostics)
