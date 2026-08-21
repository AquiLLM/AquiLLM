"""Metrics, signatures, and production-seam contracts for Task20 retrieval eval."""

from __future__ import annotations

import inspect
from hashlib import sha256
from types import SimpleNamespace

import pytest

from apps.documents.services import chunk_search
from apps.knowledge_graph.evals import run_kg_eval
from apps.knowledge_graph.retrieval.types import GraphExpansionRequest
from apps.knowledge_graph.tests import retrieval_eval_support as support


def test_ranked_metrics_cover_quality_security_latency_citations_and_distance_two():
    metrics = run_kg_eval.score_ranked_retrieval(
        expected_chunk_ids=(2, 4),
        ranked_chunk_ids=(1, 2, 3, 4, 99),
        accessible_chunk_ids={1, 2, 3, 4},
        graph_chunk_ids=(3, 4),
        citation_evidence_chunk_ids={1, 2, 3, 4},
        seed_chunk_ids=(1, 2),
        mapped_seed_chunk_ids={1},
        semantic_distances={3: 1, 4: 2},
        latency_ms=17.5,
        node_count=7,
        edge_count=5,
    )

    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr"] == 0.5
    assert 0.0 < metrics["ndcg_at_10"] <= 1.0
    assert metrics["graph_hit_rate"] == 1.0
    assert metrics["inaccessible_result_count"] == 1
    assert metrics["latency_ms"] == 17.5
    assert metrics["citation_evidence_coverage"] == 0.8
    assert metrics["seed_coverage"] == 0.5
    assert metrics["node_count"] == 7 and metrics["edge_count"] == 5
    assert metrics["distance_2_novel_fraction"] == 0.5


@pytest.mark.parametrize(
    "field",
    (
        "scope",
        "candidates",
        "seeds",
        "artifact_versions",
        "canonical_memberships",
        "mention_identity_mappings",
        "nodes",
        "pre_normalized_groups",
        "relations",
        "evidence",
    ),
)
def test_comparison_signature_changes_for_every_authorized_snapshot_component(field):
    base = {
        name: [[name, "base"]]
        for name in (
            "scope",
            "candidates",
            "seeds",
            "artifact_versions",
            "canonical_memberships",
            "mention_identity_mappings",
            "nodes",
            "pre_normalized_groups",
            "relations",
            "evidence",
        )
    }
    changed = {key: list(value) for key, value in base.items()}
    changed[field] = [[field, "changed"]]

    assert run_kg_eval.comparison_snapshot_signature(base) != (
        run_kg_eval.comparison_snapshot_signature(changed)
    )


def test_comparison_signature_changes_when_only_fallback_mention_identity_changes():
    scope, candidates, graph = support.core_fixture()
    evidence = SimpleNamespace(
        chunk_id=1,
        document_id=support.DOC_A,
        chunk_number=0,
        confidence=0.9,
        provenance_key="mention:1",
    )
    request = GraphExpansionRequest(
        seeds=candidates.graph_seeds,
        allowed_doc_ids=scope.allowed_doc_ids,
        allowed_collection_ids=scope.allowed_collection_ids,
    )

    def signature(identity_key):
        projected = SimpleNamespace(
            **{
                **vars(graph),
                "mentions": (
                    SimpleNamespace(evidence=evidence, identity_key=identity_key),
                ),
            }
        )
        payload = run_kg_eval._comparison_signature_payload(
            scope=scope,
            candidate_snapshot=candidates,
            request=request,
            graph_snapshot=projected,
        )
        return run_kg_eval.comparison_snapshot_signature(payload)

    assert signature(("fallback", 1)) != signature(("fallback", 2))


def test_eval_runner_reuses_shipping_search_and_ranking_without_duplication():
    production_source = inspect.getsource(chunk_search.text_chunk_search)
    live_source = inspect.getsource(run_kg_eval._live_comparison_bundle)
    comparison_source = inspect.getsource(run_kg_eval.run_one_snapshot_comparison)

    assert production_source.count("collect_hybrid_candidate_snapshot(") == 1
    assert "materialize_and_rerank_candidates(" in production_source
    assert "collect_hybrid_candidate_snapshot" in live_source
    assert "load_authorized_graph_snapshot" in live_source
    assert "rank_authorized_graph_snapshot" in live_source
    assert "materialize_and_rerank_candidates" in live_source
    assert "L2Distance" not in comparison_source
    assert "TrigramSimilarity" not in comparison_source


def test_fixture_checksum_is_stable_and_covers_both_fixture_files(tmp_path):
    extraction = tmp_path / "extraction.yaml"
    retrieval = tmp_path / "retrieval.yaml"
    extraction.write_bytes(b"extraction\n")
    retrieval.write_bytes(b"retrieval\n")

    expected = sha256(
        b"extraction_cases.yaml\x00extraction\n\x00retrieval_cases.yaml\x00retrieval\n"
    ).hexdigest()
    observed = run_kg_eval.fixture_checksum(extraction, retrieval)

    assert observed == expected
    assert run_kg_eval.fixture_checksum(extraction, retrieval) == observed


TASK23_ARMS = (
    "vector_only",
    "direct",
    "extended",
    "combined",
    "combined_reranked",
)


def task23_inputs():
    case = {
        "id": "task23-case",
        "accessible_collection_ids": ["collection-a"],
        "documents": [
            {
                "doc_id": "document-a",
                "collection_id": "collection-a",
                "chunks": [
                    {"chunk_id": "seed", "text": "Atlas is evaluated."},
                    {"chunk_id": "answer", "text": "Atlas uses nDCG at ten."},
                ],
            }
        ],
        "expected_retrieval_chunk_ids": ["answer"],
        "expected_min_semantic_distance": {"answer": 2},
        "quality_tags": ["relationship", "two_hop"],
    }
    observations = {}
    for index, arm in enumerate(TASK23_ARMS, start=1):
        graph = [] if arm == "vector_only" else ["answer"]
        ranked = ["seed"] if arm == "vector_only" else ["seed", "answer"]
        if arm == "combined_reranked":
            ranked = ["answer", "seed"]
        observations[arm] = [
            {
                "case_id": "task23-case",
                "ranked_chunk_ids": ranked,
                "graph_chunk_ids": graph,
                "citation_evidence_chunk_ids": ranked,
                "seed_chunk_ids": ["seed"],
                "mapped_seed_chunk_ids": ["seed"],
                "projected_ranks": graph,
                "repeated_projected_ranks": graph,
                "adversarial_candidate_chunk_ids": ["private"],
                "inaccessible_result_chunk_ids": [],
                "latency_ms": float(index),
                "reranker_calls": 1 if arm == "combined_reranked" else 0,
                "comparison_snapshot_signature": "a" * 64,
            }
        ]
    freshness = {
        "generation_key": "b" * 64,
        "projection_checksum": "c" * 64,
        "age_seconds": 5.0,
        "max_age_seconds": 60.0,
    }
    parity = {
        f"{backend}_{kind}_sha256": character * 64
        for kind, character in (
            ("snapshot", "d"),
            ("scores", "e"),
            ("trace", "f"),
            ("ties", "1"),
        )
        for backend in ("postgres", "memgraph")
    }
    parity.update(
        postgres_projected_ranks=["entity-a", "entity-b"],
        memgraph_projected_ranks=["entity-a", "entity-b"],
    )
    return (case,), observations, freshness, parity


def test_task23_metrics_cover_all_five_arms_quality_multihop_latency_and_citations():
    cases, observations, freshness, parity = task23_inputs()

    report = run_kg_eval.build_task21_hybrid_report(
        cases=cases,
        observations=observations,
        freshness=freshness,
        backend_parity=parity,
    )

    assert tuple(report["arms"]) == TASK23_ARMS
    assert report["arms"]["vector_only"]["metrics"]["recall_at_10"] == 0.0
    assert report["arms"]["combined_reranked"]["metrics"]["ndcg_at_10"] == 1.0
    assert report["arms"]["extended"]["metrics"]["distance_2_novel_fraction"] == 1.0
    assert report["arms"]["combined"]["metrics"]["citation_evidence_coverage"] == 1.0
    assert report["arms"]["combined_reranked"]["metrics"]["latency_p95_ms"] == 5.0
    assert report["arms"]["combined"]["metrics"]["adversarial_candidate_count"] == 1
    assert report["observed_inaccessible_result_count"] == 0
    assert report["observed_adversarial_candidate_count"] == len(TASK23_ARMS)
    assert report["permission_isolation"] is True
