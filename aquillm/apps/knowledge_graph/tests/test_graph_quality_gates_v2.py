"""Absolute identity, unique paired cohorts, uncertainty and fail-closed inputs."""

import importlib
from copy import deepcopy

import pytest

from apps.knowledge_graph.tests.graph_gate_test_fixtures import fixture_report, seal


def evaluate(report):
    try:
        module = importlib.import_module(
            "apps.knowledge_graph.evals.graph_quality_gates_v2"
        )
    except ModuleNotFoundError:
        pytest.fail("G6 absolute graph gates are missing")
    return module.evaluate_graph_gates_v2(
        report=report, policy=module.GraphGatePolicyV2()
    )


def test_synthetic_positive_proves_predicates_only():
    report = fixture_report()
    result = evaluate(report)
    assert result["eligible"], result
    assert result["identity"]["automatic_total"] == 600
    assert result["retrieval"]["query_pairs"] == 150
    report["backend"] = "synthetic"
    assert "synthetic_evidence" in evaluate(report)["blocking_reasons"]


@pytest.mark.parametrize(
    "correct,total,reason",
    [
        (60, 600, "automatic_precision_below_floor"),
        (1, 1, "automatic_pairs_insufficient"),
        (597, 600, "automatic_precision_interval_below_floor"),
    ],
)
def test_relative_precision_or_small_sample_cannot_pass(correct, total, reason):
    report = fixture_report()
    report["decisions"] = report["decisions"][:total]
    for i, row in enumerate(report["decisions"]):
        row["expected_same"] = i < correct
    result = evaluate(seal(report))
    assert not result["eligible"]
    assert reason in result["blocking_reasons"]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_decision",
        "duplicate_query",
        "missing_label",
        "bool_metric",
        "known_conflict",
        "missing_identity_stratum",
        "missing_query_stratum",
        "budget_drift",
        "runtime_drift",
        "leaked_labels",
        "unknown_cache",
        "bad_coverage",
    ],
)
def test_invalid_or_missing_observations_fail(mutation):
    report = fixture_report()
    query = report["queries"][0]
    if mutation == "duplicate_decision":
        report["decisions"].append(deepcopy(report["decisions"][0]))
    elif mutation == "duplicate_query":
        report["queries"].append(deepcopy(query))
    elif mutation == "missing_label":
        del query["labels"]["no_answer"]
    elif mutation == "bool_metric":
        query["executions"]["baseline"]["candidate"]["latency_ms"] = True
    elif mutation == "known_conflict":
        report["decisions"][-1]["automatic"] = True
    elif mutation == "missing_identity_stratum":
        for row in report["decisions"]:
            row["strata"] = ["stable_identifiers"]
    elif mutation == "missing_query_stratum":
        for row in report["queries"]:
            row["strata"] = ["homonym"]
    elif mutation.endswith("_drift"):
        query["executions"]["baseline"]["candidate"]["binding"][
            mutation.split("_")[0]
        ] = "f" * 64
    elif mutation == "leaked_labels":
        query["executions"]["baseline"]["candidate"]["inputs"]["labels"] = query[
            "labels"
        ]
    elif mutation == "unknown_cache":
        del query["executions"]["baseline"]["candidate"]["provider_cache"]
    else:
        query["executions"]["baseline"]["candidate"]["coverage"] = "complete"
    assert not evaluate(seal(report))["eligible"]


def test_arbitrary_chunks_do_not_score_no_answer():
    report = fixture_report()
    row = report["queries"][0]
    row["labels"] = dict(relevant=[], no_answer=True, coverage="partial")
    result = evaluate(seal(report))
    assert not result["eligible"]
    assert "no_answer_failed" in result["blocking_reasons"]


def test_one_all_tag_gain_cannot_hide_regressions():
    report = fixture_report()
    for i, row in enumerate(report["queries"]):
        if row["labels"]["no_answer"]:
            continue
        row["executions"]["baseline"]["baseline"]["ranked"] = row["executions"][
            "baseline"
        ]["baseline"]["repeated_ranked"] = ["good"]
        row["executions"]["baseline"]["candidate"]["ranked"] = row["executions"][
            "baseline"
        ]["candidate"]["repeated_ranked"] = ["bad"]
    row = report["queries"][0]
    from apps.knowledge_graph.tests.graph_gate_test_fixtures import STRATA

    row["strata"] = list(STRATA)
    row["executions"]["baseline"]["baseline"]["ranked"] = row["executions"]["baseline"][
        "baseline"
    ]["repeated_ranked"] = ["bad"]
    row["executions"]["baseline"]["candidate"]["ranked"] = row["executions"][
        "baseline"
    ]["candidate"]["repeated_ranked"] = ["good"]
    result = evaluate(seal(report))
    assert "aggregate_retrieval_regression" in result["blocking_reasons"]


def test_intersection_regression_cannot_hide_behind_stratum_marginals():
    report = fixture_report()
    # Every marginal has gains; the two-tag intersection has only losses.
    from apps.knowledge_graph.tests.graph_gate_test_fixtures import STRATA, digest

    report["queries"] = [deepcopy(report["queries"][0]) for _ in range(312)]
    for i, row in enumerate(report["queries"]):
        row["query_id"] = f"cohort-{i}"
        for mode, pair in row["executions"].items():
            for arm, execution in pair.items():
                execution["inputs"].update(
                    query_id=row["query_id"], question=f"cohort question {i}"
                )
                execution["frontier_invocations"][0].update(
                    invocation_id=f"cohort-{i}-{mode}-{arm}",
                    query_sha256=digest(execution["inputs"]["question"]),
                )
        row["strata"] = (
            [STRATA[i % 6]] if i < 300 or i >= 306 else [STRATA[0], STRATA[1]]
        )
        row["executions"]["baseline"]["candidate"]["ranked"] = row["executions"][
            "baseline"
        ]["candidate"]["repeated_ranked"] = ["good"] if i < 300 else ["bad"]
        row["executions"]["baseline"]["baseline"]["ranked"] = row["executions"][
            "baseline"
        ]["baseline"]["repeated_ranked"] = ["bad"] if i < 300 else ["good"]
        if i >= 306:
            row["labels"].update(relevant=[], no_answer=True)
            for pair in row["executions"].values():
                for execution in pair.values():
                    execution.update(ranked=[], repeated_ranked=[], abstained=True)
    result = evaluate(seal(report))
    assert not any("insufficient" in r for r in result["blocking_reasons"])
    cohorts = result["retrieval"]["modes"]["baseline"]["cohorts"]
    assert cohorts["aggregate"]["recall_at_10"]["lower"] > 0
    assert all(cohorts[s]["recall_at_10"]["lower"] > 0 for s in STRATA)
    assert any("homonym&relation_intent" in r for r in result["blocking_reasons"])


def test_legacy_explicitly_ineligible():
    result = evaluate({"auto_link_precision": 1})
    assert result["eligible"] is False
    assert "legacy_report_ineligible" in result["blocking_reasons"]
