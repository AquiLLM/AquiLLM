"""Evaluator catalogs must never promote requested controls to observed evidence."""

import json
from hashlib import sha256

import pytest


def build(**kwargs):
    from apps.chat.evals.retrieval_treatment_manifest import treatment_manifest

    return treatment_manifest(**kwargs)


def test_catalog_covers_independent_dimensions_without_default_observations():
    result = build()
    assert result["schema_version"] == 1
    assert set(result["treatments"]) == {
        "window_source",
        "retrieval_fusion",
        "query_proposals",
        "candidate_depth",
        "structure",
        "embedding_preparation",
        "graph",
        "document_capacity",
        "followup",
        "iterative",
        "adaptive_selection",
        "reranker_model",
    }
    assert result["reports"] == {}
    assert result["activation_decision"] == "not_issued_by_catalog"
    for row in result["treatments"].values():
        assert row["configured"] is None
        assert row["report_ids"] == []
        assert row["result"] == "inconclusive"
        assert row["dependency_observations"] == {}


def test_configured_identities_never_fill_missing_runtime_or_observations():
    configured = {"window_source": {"mode": "source", "runtime_sha256": "a" * 64}}
    result = build(configured=configured, dependencies={"I5": "pending_review"})
    row = result["treatments"]["window_source"]
    assert row["configured"] == configured["window_source"]
    assert row["result"] == "inconclusive" and not result["reports"]
    assert result["treatments"]["structure"]["dependency_observations"] == {
        "I5": "pending_review"
    }
    row["configured"]["mode"] = "mutated"
    assert configured["window_source"]["mode"] == "source"


def test_original_report_bytes_gate_and_observed_rows_remain_distinct():
    observed = {"final_passages": 5, "candidates_per_action": 15}
    gate = {"activation_eligible": False, "blocking_reasons": ["missing human"]}
    body = {
        "backend": "fixture",
        "quality_claim": "unmeasured",
        "activation_eligible": False,
        "gate": gate,
        "observations": [{"case_id": "public-1", "comparison_controls": observed}],
    }
    raw = (json.dumps(body, indent=2) + "\n").encode()
    result = build(
        configured={"candidate_depth": {"candidates_per_action": 45}},
        reports={"original.json": raw},
        associations={"candidate_depth": ["original.json"]},
    )
    report = result["reports"]["original.json"]
    assert report["sha256"] == sha256(raw).hexdigest()
    assert report["gate"] == gate
    assert report["observations"] == body["observations"]
    assert "runtime_verified" not in report
    assert result["treatments"]["candidate_depth"]["result"] == "inconclusive"
    assert result["treatments"]["structure"]["report_ids"] == []


def test_catalog_preserves_claims_without_issuing_activation_or_trust():
    raw = json.dumps({"backend": "fixture", "activation_eligible": True}).encode()
    result = build(reports={"untrusted.json": raw})
    assert result["reports"]["untrusted.json"]["activation_eligible"] is True
    assert result["activation_decision"] == "not_issued_by_catalog"
    assert all(row["result"] == "inconclusive" for row in result["treatments"].values())


@pytest.mark.parametrize(
    "arguments",
    [
        {"configured": {"unknown_treatment": {}}},
        {"associations": {"window_source": ["missing.json"]}},
        {"reports": {"broken.json": b"not JSON"}},
    ],
)
def test_unknown_or_malformed_catalog_inputs_are_not_silently_completed(arguments):
    with pytest.raises(ValueError):
        build(**arguments)
