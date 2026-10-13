"""Every retained mode/arm must obey its captured effective graph deadline."""

from copy import deepcopy

import pytest

from apps.knowledge_graph.evals.graph_activation_attachment_v2 import (
    graph_activation_attachment_v2,
    validate_graph_activation_attachment_v2,
)
from apps.knowledge_graph.evals.graph_gate_observations import digest
from apps.knowledge_graph.tests.graph_review_fixtures import MODES
from apps.knowledge_graph.tests.test_graph_gate_applicability import attachment_pair

REASON = "invalid_or_missing_observation:graph_execution_exceeds_effective_deadline"


def repin(report, quality):
    """Recompute originals, including q0 pins; no stale-digest rejection suffices."""
    attached = graph_activation_attachment_v2(report)
    queries = {q["query_id"]: q for q in report["queries"]}
    for mode, evidence in quality.items():
        for row in evidence["observations"]:
            graph = row["lane_observations"]["graph"]
            query = queries[graph["query_id"]]
            graph.update(
                attachment_sha256=attached["attachment_sha256"],
                observation_sha256=digest(query),
                execution_sha256=digest(query["executions"][mode]["candidate"]),
            )
    return attached


def test_rehashed_unreferenced_query_cannot_enlarge_actual_deadline(monkeypatch):
    report, quality = attachment_pair(monkeypatch)
    controls = deepcopy(report["controls"])
    bindings = deepcopy((report["binding"], report["baseline_binding"]))
    assert all(
        c["runtime_observations"]["graph_overall_timeout_ms"] == 300
        for c in controls.values()
    )
    for pair in report["queries"][1]["executions"].values():
        for execution in pair.values():
            execution.update(latency_budget_ms=1000, latency_ms=500)
    attached = repin(report, quality)
    assert report["controls"] == controls
    assert (report["binding"], report["baseline_binding"]) == bindings
    assert all(
        q["observations"][0]["lane_observations"]["graph"]["query_id"] == "q0"
        for q in quality.values()
    )
    result = validate_graph_activation_attachment_v2(report, quality_reports=quality)
    assert not result["passed"], result
    assert REASON in result["blocking_reasons"]
    assert not attached["result"]["eligible"]


@pytest.mark.parametrize("query_index", [0, 1])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("arm", ["baseline", "candidate"])
def test_each_retained_arm_mode_checks_its_actual_upper_bound(
    monkeypatch, query_index, mode, arm
):
    report, quality = attachment_pair(monkeypatch)
    execution = report["queries"][query_index]["executions"][mode][arm]
    actual = report["controls"][arm]["runtime_observations"]["graph_overall_timeout_ms"]
    execution.update(latency_budget_ms=actual + 1, latency_ms=actual + 1)
    repin(report, quality)
    result = validate_graph_activation_attachment_v2(report, quality_reports=quality)
    assert not result["passed"]
    assert REASON in result["blocking_reasons"], result


@pytest.mark.parametrize("remaining_budget", [50, 300])
def test_equal_or_stricter_observed_allowances_remain_valid(
    monkeypatch, remaining_budget
):
    report, quality = attachment_pair(monkeypatch)
    for query in report["queries"]:
        for pair in query["executions"].values():
            for execution in pair.values():
                execution.update(
                    latency_budget_ms=remaining_budget, latency_ms=remaining_budget
                )
    repin(report, quality)
    result = validate_graph_activation_attachment_v2(report, quality_reports=quality)
    assert result["passed"], result
