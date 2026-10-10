"""Boundary values, held-out leakage and uncertainty cannot silently pass."""

from copy import deepcopy
from dataclasses import replace

import pytest

from apps.knowledge_graph.evals.graph_quality_gates_v2 import (
    GraphGatePolicyV2,
    evaluate_graph_gates_v2,
)
from apps.knowledge_graph.evals.retrieval_eval_statistics import (
    paired_graph_interval,
    wilson_95,
)
from apps.knowledge_graph.tests.graph_gate_test_fixtures import (
    digest,
    fixture_report,
    seal,
)


def evaluate(report):
    return evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())


def test_precise_wilson_two_sided_and_unknown():
    assert wilson_95(600, 600)["lower"] == pytest.approx(0.993638298986299)
    assert wilson_95(597, 600)["lower"] < 0.99
    assert wilson_95(0, 0) == {"lower": None, "upper": None}
    with pytest.raises(ValueError):
        wilson_95(True, 1)


def test_relabelled_same_pair_and_query_cannot_inflate_sample():
    report = fixture_report()
    report["decisions"][1]["pair_ids"] = list(
        reversed(report["decisions"][0]["pair_ids"])
    )
    assert (
        "invalid_or_missing_observation:duplicate_or_invalid_pair"
        in evaluate(seal(report))["blocking_reasons"]
    )
    report = fixture_report()
    for arm in ("baseline", "candidate"):
        report["queries"][1][arm]["inputs"].update(
            question=report["queries"][0][arm]["inputs"]["question"],
            scope_sha256=report["queries"][0][arm]["inputs"]["scope_sha256"],
        )
        report["queries"][1][arm]["frontier_invocations"][0]["query_sha256"] = digest(
            report["queries"][1][arm]["inputs"]["question"]
        )
    assert (
        "invalid_or_missing_observation:duplicate_query_content"
        in evaluate(report)["blocking_reasons"]
    )


def test_relative_ten_percent_vs_one_percent_still_fails_absolute_floor():
    report = fixture_report()
    for i, row in enumerate(report["decisions"][:600]):
        row["expected_same"] = i < 60
        row["candidate"] = i < 6 or i >= 60
    report["decisions"][-1]["candidate"] = False
    for i in range(54):
        report["decisions"].append(
            dict(
                decision_id=f"extra-{i}",
                pair_ids=[f"x{i}", f"y{i}"],
                strata=["alias_collisions"],
                automatic=False,
                candidate=True,
                expected_same=False,
                known_conflict=False,
            )
        )
    result = evaluate(seal(report))
    assert result["identity"]["automatic_precision"] == 0.1
    assert result["identity"]["candidate_precision"] == 0.01
    assert "automatic_precision_below_floor" in result["blocking_reasons"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_automatic_precision", 0.1),
        ("max_metric_decline", False),
        ("min_query_pairs", 1),
    ],
)
def test_thresholds_cannot_be_weakened_after_measurement(field, value):
    policy = replace(GraphGatePolicyV2(), **{field: value})
    assert (
        "unrecognized_graph_policy"
        in evaluate_graph_gates_v2(report=fixture_report(), policy=policy)[
            "blocking_reasons"
        ]
    )


def test_zero_deltas_cannot_prove_targeted_improvement():
    report = fixture_report()
    for row in report["queries"]:
        row["baseline"] = deepcopy(row["candidate"])
        row["baseline"]["frontier_invocations"][0]["invocation_id"] = (
            row["query_id"] + "-baseline"
        )
    result = evaluate(report)
    assert not result["eligible"]
    assert any("intended_improvement_unproven" in r for r in result["blocking_reasons"])
    assert paired_graph_interval([0] * 150)["lower"] == 0


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
def test_bounded_graph_cannot_claim_exhaustive_or_contradict_abstention(arm):
    for field in ("exhaustive", "abstained"):
        report = fixture_report()
        report["queries"][0][arm][field] = True
        assert not evaluate(report)["eligible"]


@pytest.mark.parametrize(
    "mutation",
    [
        "nan",
        "negative",
        "unknown_boolean",
        "late_policy",
        "heldout_query",
        "heldout_decision",
        "missing_conflicts",
        "source",
        "model",
        "budget",
        "labels",
        "candidate_count",
        "completion",
        "determinism",
        "safety",
        "latency",
    ],
)
def test_malformed_and_unmatched_study_values_block(mutation):
    report = fixture_report()
    row = report["queries"][0]
    if mutation == "nan":
        row["candidate"]["latency_ms"] = float("nan")
    elif mutation == "negative":
        row["candidate"]["latency_ms"] = -1
    elif mutation == "unknown_boolean":
        report["decisions"][0]["expected_same"] = None
    elif mutation == "late_policy":
        report["predeclared_at"] = report["heldout_opened_at"] + 1
    elif mutation == "heldout_query":
        report["development_query_ids"] = [row["query_id"]]
    elif mutation == "heldout_decision":
        report["development_decision_ids"] = [report["decisions"][0]["decision_id"]]
    elif mutation == "missing_conflicts":
        report["decisions"] = report["decisions"][:-1]
        seal(report)
    elif mutation in ("source", "model", "budget"):
        field = {
            "source": "source_generation",
            "model": "model_runtime",
            "budget": "budget",
        }[mutation]
        report["baseline_binding"][field] = "f" * 64
    elif mutation == "labels":
        report["labels_sha256"] = "f" * 64
    elif mutation == "candidate_count":
        report["decisions"][0]["candidate"] = None
    elif mutation == "completion":
        row["candidate"]["completion"] = "cancelled"
    elif mutation == "determinism":
        row["candidate"]["repeated_ranked"] = ["other"]
    elif mutation == "safety":
        row["candidate"]["authorization_violations"] = 1
    else:
        row["candidate"]["latency_ms"] = 21
    assert not evaluate(report)["eligible"]
