"""Regression checks for one vote per query and closed score sidecars."""

import pytest

from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
from apps.documents.services.chunk_rerank_results import RerankScoreSet
from apps.documents.services.chunk_rerank_score_transport import deserialize_score_set


def row(pk, rank):
    return {
        "chunk_id": pk,
        "doc_id": "a",
        "chunk": pk - 1,
        "rank": rank,
        "text": f"Passage {pk}",
        "citation": f"[doc:a chunk:{pk}]",
    }


def test_one_rrf_vote_per_query_but_distinct_queries_accumulate():
    pool = fuse_ranked_tool_results(
        [
            {"result": [row(1, 1), row(2, 2), row(2, 2)]},
            {"result": [row(1, 1)]},
        ]
    )
    scores = dict(pool.fused_scores)
    assert scores["[doc:a chunk:1]"] == pytest.approx(2 / 61)
    assert scores["[doc:a chunk:2]"] == pytest.approx(1 / 62)
    assert [item["chunk_id"] for item in pool.rows] == [1, 2]


def test_transport_rejects_complete_rank_only_sidecar():
    sidecar = {
        "schema_version": "v2",
        "query_fingerprint": "query",
        "scorer_fingerprint": "scorer",
        "pool_fingerprint": "pool",
        "scoring_kind": "rank_only",
        "status": "complete",
        "candidate_order": [1],
        "scores": [],
    }
    assert deserialize_score_set(sidecar) is None


def test_typed_sidecar_cannot_bypass_status_validation():
    malformed = RerankScoreSet(
        "v2",
        "query",
        "scorer",
        "pool",
        "rank_only",
        "complete",
        (1,),
        (),
    )
    assert deserialize_score_set(malformed) is None
