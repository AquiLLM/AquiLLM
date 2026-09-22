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


def test_merge_keeps_second_document_before_final_cutoff(monkeypatch):
    monkeypatch.setenv("RAG_MAX_SNIPPETS_PER_DOC", "2")
    merged = _merge([
        _payload(_row(1, 1, "a"), _row(2, 2, "a"), _row(3, 3, "b")),
    ], limit=2)

    assert [row["chunk_id"] for row in merged["result"]] == [1, 3]
    assert [row["rank"] for row in merged["result"]] == [1, 2]


def test_merge_caps_each_document_and_keeps_its_ranking(monkeypatch):
    monkeypatch.setenv("RAG_MAX_SNIPPETS_PER_DOC", "2")
    merged = _merge([
        _payload(*[_row(i, i, "a") for i in range(1, 5)], _row(5, 5, "b")),
    ], limit=5)

    assert [row["chunk_id"] for row in merged["result"]] == [1, 5, 2]
    assert merged["retrieved_count"] == 3


def test_merge_titles_describe_only_retained_rows(monkeypatch):
    merged = _merge([_payload(_row(1, 1, "a"), _row(2, 2, "b"))], limit=1)
    assert merged["retrieved_documents"] == ["Paper a"]


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


def test_fusion_does_not_make_the_final_selection():
    from apps.chat.services.rag_retrieval import fuse_ranked_tool_results

    rows = [
        {"rank": 1, "chunk_id": 1, "doc_id": "a", "chunk": 0,
         "text": "Primary evidence", "citation": "[doc:a chunk:1]"},
        {"rank": 2, "chunk_id": 2, "doc_id": "a", "chunk": 1,
         "text": "Complementary evidence", "citation": "[doc:a chunk:2]"},
        {"rank": 3, "chunk_id": 3, "doc_id": "b", "chunk": 0,
         "text": "Secondary evidence", "citation": "[doc:b chunk:3]"},
    ]
    pool = fuse_ranked_tool_results([{"result": rows}], candidate_limit=45)
    assert [row["chunk_id"] for row in pool.rows] == [1, 2, 3]


def test_fusion_rejects_inconsistent_coordinates_and_citations():
    from apps.chat.services.rag_retrieval import fuse_ranked_tool_results

    good = {**_row(1, 1, "a"), "chunk": 0}
    invalid = {**_row(2, 2, "b"), "chunk": 0,
               "citation": "[doc:b chunk:999]"}
    collision = {**_row(1, 1, "a"), "chunk": 5}
    pool = fuse_ranked_tool_results(
        [_payload(good, invalid), _payload(collision)], candidate_limit=45
    )
    assert pool.rows == (good,)


def test_fusion_keeps_per_query_scores_separate_and_caps_union():
    from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
    from apps.documents.services.chunk_rerank_results import PassageScore, RerankScoreSet
    from apps.documents.services.chunk_rerank_score_transport import serialize_score_set
    from uuid import UUID

    doc = UUID("00000000-0000-0000-0000-000000000001")
    score_set = RerankScoreSet(
        "v2", "query", "scorer", "pool", "pointwise", "complete", (1,),
        (PassageScore(1, doc, 0, "source", "effective", 0.9),),
    )
    row = _row(1, 1, str(doc))
    row["chunk"] = 0
    first = {**_payload(row), "_retrieval_scores": serialize_score_set(score_set)}
    second = {**_payload({**row, "rank": 2}),
              "_retrieval_scores": serialize_score_set(score_set)}
    pool = fuse_ranked_tool_results([first, second], candidate_limit=1)
    assert len(pool.rows) == 1
    assert pool.source_score_sets == (score_set, score_set)
    assert pool.fused_scores[0][0] == row["citation"]
    assert pool.fused_scores[0][1] > 0


def test_fusion_ignores_malformed_or_unrelated_scores_and_nested_score_fields():
    from apps.chat.services.rag_retrieval import fuse_ranked_tool_results

    row = {**_row(1, 1, "a"), "chunk": 0, "score": 0.93,
           "_score_set": {"secret": "private"}}
    pool = fuse_ranked_tool_results([
        {**_payload(row), "_retrieval_scores": {"scores": [0.93]}},
        {**_payload(row), "_retrieval_scores": {
            "schema_version": "v2", "query_fingerprint": "q",
            "scorer_fingerprint": "s", "pool_fingerprint": "p",
            "scoring_kind": "pointwise", "status": "complete",
            "candidate_order": [999], "scores": [],
        }},
    ])
    assert pool.source_score_sets == ()
    assert "score" not in pool.rows[0]
    assert "_score_set" not in pool.rows[0]
