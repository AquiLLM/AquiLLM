"""Synthetic arithmetic controls, never measured graph acceptance."""
from copy import deepcopy

import pytest

from apps.knowledge_graph.evals.graph_quality_gates_v2 import (
    GraphGatePolicyV2, evaluate_graph_gates_v2,
)
from apps.knowledge_graph.tests.graph_gate_test_fixtures import STRATA, fixture_report, seal
from apps.knowledge_graph.tests.graph_review_fixtures import MODES


def synthetic_report():
    report = fixture_report()
    # The reused builder supplies invented observations for predicate tests.
    # Never evaluate or export them with its historical live backend label.
    report["backend"] = "synthetic"
    report["queries"][0]["strata"] = list(STRATA)
    return seal(report)


def evaluate(report):
    assert report["backend"] == "synthetic"
    return evaluate_graph_gates_v2(report=seal(report), policy=GraphGatePolicyV2())


def assert_completed_fixture(report):
    assert len({row["query_id"] for row in report["queries"]}) == 150
    assert len(report["queries"]) == 150
    for row in report["queries"]:
        for pair in row["executions"].values():
            for observation in pair.values():
                assert observation["completion"] == "completed"
                assert observation["latency_ms"] <= observation["latency_budget_ms"]
                assert observation["authorization_violations"] == observation["citation_violations"] == 0


@pytest.mark.parametrize("regressed_mode", MODES)
def test_one_tagged_gain_cannot_override_broad_loss_despite_completed_fixture(regressed_mode):
    positive = synthetic_report()
    assert_completed_fixture(positive)
    control = evaluate(positive)
    assert control["eligible"] is False
    assert control["blocking_reasons"] == ["synthetic_evidence"]
    assert control["identity"]["automatic_total"] == 600
    assert control["retrieval"]["query_pairs"] == 150
    assert all(value["query_pairs"] == 150 and not value["blocking_reasons"]
               for value in control["retrieval"]["modes"].values())

    adverse = deepcopy(positive)
    for row in adverse["queries"][1:]:
        if row["labels"]["no_answer"]:
            continue
        candidate = row["executions"][regressed_mode]["candidate"]
        candidate["ranked"] = candidate["repeated_ranked"] = ["bad"]
    assert_completed_fixture(adverse)
    result = evaluate(adverse)
    assert result["eligible"] is False and "synthetic_evidence" in result["blocking_reasons"]
    assert "aggregate_retrieval_regression" in result["blocking_reasons"]
    assert not any(reason.startswith("invalid_or_missing_observation")
                   for reason in result["blocking_reasons"])
    assert result["identity"] == control["identity"]
    assert result["retrieval"]["query_pairs"] == 150
    metrics = result["retrieval"]["modes"][regressed_mode]
    gain = metrics["observations"][positive["queries"][0]["query_id"]]
    assert gain["candidate"]["recall_at_10"] > gain["baseline"]["recall_at_10"]
    assert metrics["cohorts"]["aggregate"]["recall_at_10"]["mean"] < 0
    for stratum in STRATA:
        assert metrics["cohorts"][stratum]["recall_at_10"]["mean"] < 0
        assert any(reason.startswith(f"retrieval_cohort_regression:{stratum}:")
                   for reason in metrics["blocking_reasons"])
    for mode in MODES:
        if mode != regressed_mode:
            assert result["retrieval"]["modes"][mode] == control["retrieval"]["modes"][mode]
    assert adverse["binding"] == positive["binding"]
    assert adverse["baseline_binding"] == positive["baseline_binding"]
    assert adverse["controls"] == positive["controls"]
