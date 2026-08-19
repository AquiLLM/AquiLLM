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
