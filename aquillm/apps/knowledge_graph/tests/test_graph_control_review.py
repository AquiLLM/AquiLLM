"""Fully rehashed malformed controls and opaque baseline originals must fail."""

from copy import deepcopy

import pytest

from apps.knowledge_graph.evals.graph_activation_attachment_v2 import (
    validate_graph_activation_attachment_v2,
)
from apps.knowledge_graph.evals.graph_effective_controls import graph_effective_controls
from apps.knowledge_graph.evals.graph_gate_observations import digest
from apps.knowledge_graph.evals.graph_quality_gates_v2 import (
    GraphGatePolicyV2,
    evaluate_graph_gates_v2,
)
from apps.knowledge_graph.tests.graph_review_fixtures import (
    originals_report,
    rebind,
    rehash,
)
from apps.knowledge_graph.tests.test_graph_gate_applicability import attachment_pair


@pytest.mark.parametrize(
    "field,value",
    [
        ("graph_direct_max_nodes", None),
        ("graph_direct_max_nodes", True),
        ("graph_direct_max_nodes", "200"),
        ("graph_direct_max_nodes", -1),
        ("graph_direct_max_nodes", 201),
        ("graph_direct_enabled", "unknown"),
        ("graph_direct_timeout_ms", -1),
        ("graph_direct_timeout_ms", 5001),
        ("direct_min_similarity", True),
        ("direct_winner_margin", 0.99),
        ("graph_topology_backend", "unknown"),
        ("query_extractor_model_revision", "unknown"),
    ],
)
def test_rehashed_invalid_controls_reject_required_attachment(
    monkeypatch, field, value
):
    control = graph_effective_controls()
    control["remediation_mode"] = "v2"
    control["runtime_observations"][field] = value
    rehash(control)
    report, quality = attachment_pair(monkeypatch, control)
    assert not validate_graph_activation_attachment_v2(report, quality_reports=quality)[
        "passed"
    ]


@pytest.mark.parametrize(
    "mutation",
    ["unknown_hash", "unknown_config", "model_drift", "cap_drift", "honest_cap_drift"],
)
def test_baseline_original_controls_are_verified(mutation):
    report = originals_report()
    base = report["controls"]["baseline"]
    if mutation == "unknown_hash":
        base["algorithm_signature"] = "a" * 64
        base["runtime_sha256"] = digest(
            {k: v for k, v in base.items() if k != "runtime_sha256"}
        )
    elif mutation == "unknown_config":
        base["algorithm_config"]["canonical_resolver_version"] = "invented"
        rehash(base)
    elif mutation == "model_drift":
        base["runtime_observations"]["query_extractor_model_revision"] = "b" * 40
        rehash(base)
    else:
        base["runtime_observations"]["graph_direct_max_nodes"] -= 1
        rehash(base)
    rebind(report)
    if mutation == "cap_drift":
        # Even copying the candidate budget cannot validate a different original cap.
        report["baseline_binding"]["graph_budget"] = report["binding"]["graph_budget"]
        for query in report["queries"]:
            for pair in query["executions"].values():
                pair["baseline"]["binding"] = deepcopy(report["baseline_binding"])
    assert not evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())[
        "eligible"
    ]


def test_stricter_actual_controls_preserve_valid_required_attachment(monkeypatch):
    from django.conf import settings

    monkeypatch.setattr(settings, "KG_GRAPH_DIRECT_MAX_NODES", 150)
    monkeypatch.setattr(settings, "KG_GRAPH_DIRECT_TIMEOUT_MS", 50)
    report, quality = attachment_pair(monkeypatch)
    result = validate_graph_activation_attachment_v2(report, quality_reports=quality)
    assert result["passed"], result


@pytest.mark.parametrize("declaration", ["extra", "missing", "forbidden"])
def test_declared_treatment_cannot_waive_unrelated_control_drift(declaration):
    report = originals_report()
    if declaration == "missing":
        report["treatment"] = {}
    elif declaration == "extra":
        report["treatment"]["identity_policy"] = {
            "baseline": "legacy",
            "candidate": "legacy",
        }
    else:
        base = report["controls"]["baseline"]
        before = base["runtime_observations"]["query_extractor_model_revision"]
        base["runtime_observations"]["query_extractor_model_revision"] = "b" * 40
        rehash(base)
        rebind(report)
        report["treatment"]["runtime_observations.query_extractor_model_revision"] = {
            "baseline": "b" * 40,
            "candidate": before,
        }
    assert not evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())[
        "eligible"
    ]


@pytest.mark.parametrize("contradiction", [False, True])
@pytest.mark.parametrize(
    "field,limit,value",
    [
        ("nodes", "max_identities", 100),
        ("edges", "max_edges", 200),
        ("depth", "max_depth", 1),
    ],
)
def test_actual_enabled_caps_bound_frontier_originals(
    contradiction, field, limit, value
):
    report = originals_report()
    for control in report["controls"].values():
        control["runtime_observations"].update(
            graph_extended_enabled=True, memgraph_traversal_enabled=True
        )
        control["runtime_observations"][f"graph_extended_max_{field}"] = value
        rehash(control)
    if not contradiction:
        for query in report["queries"]:
            for pair in query["executions"].values():
                for execution in pair.values():
                    execution["frontier_budget"][limit] = value
                    execution["frontier_invocations"][0]["frontier_budget"][limit] = (
                        value
                    )
    rebind(report)
    result = evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())
    assert result["eligible"] is not contradiction, result
    if contradiction:
        assert (
            "invalid_or_missing_observation:graph_frontier_exceeds_effective_caps"
            in result["blocking_reasons"]
        )


def test_result_count_cannot_exceed_active_branch_cap():
    report = originals_report()
    for control in report["controls"].values():
        control["runtime_observations"].update(
            graph_extended_enabled=True,
            memgraph_traversal_enabled=True,
            graph_extended_max_candidates=1,
        )
        rehash(control)
    rebind(report)
    result = evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())
    assert (
        "invalid_or_missing_observation:graph_results_exceed_effective_cap"
        in result["blocking_reasons"]
    )
