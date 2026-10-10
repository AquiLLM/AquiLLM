"""Synthetic accounting originals: never clamp excess or infer unknown work."""

from copy import deepcopy

import pytest

from apps.knowledge_graph.evals.graph_quality_gates_v2 import (
    GraphGatePolicyV2,
    evaluate_graph_gates_v2,
)
from apps.knowledge_graph.tests.graph_gate_test_fixtures import fixture_report


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "overshoot",
        "uncertain",
        "incomplete",
        "forged_caps",
        "bool",
        "different_limits",
        "identities_cap",
        "edges_cap",
    ],
)
def test_frontier_unknown_or_overrun_blocks(mutation):
    report = fixture_report()
    row = report["queries"][0]["candidate"]
    observation = row["frontier_budget"]
    if mutation == "missing":
        del row["frontier_budget"]
    elif mutation == "overshoot":
        observation["bytes_used"] = observation["max_bytes"] + 1
    elif mutation == "uncertain":
        observation["uncertain"] = True
    elif mutation == "incomplete":
        observation["accounting_complete"] = False
    elif mutation == "forged_caps":
        observation["max_queries"] = 65
    elif mutation == "bool":
        observation["rows_used"] = True
    elif mutation == "identities_cap":
        observation["max_identities"] = 201
    elif mutation == "edges_cap":
        observation["max_edges"] = 1001
    else:
        observation["max_rows"] -= 1
        row["frontier_invocations"][0]["frontier_budget"] = deepcopy(observation)
    original = deepcopy(report)
    result = evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())
    assert not result["eligible"], result
    assert report == original


def test_distinct_graph_caps_cannot_hide_in_declared_policy_treatment():
    report = fixture_report()
    report["baseline_binding"]["graph_budget"] = "f" * 64
    assert not evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())[
        "eligible"
    ]


@pytest.mark.parametrize("mutation", ["zero", "two", "reuse", "query", "subset"])
def test_all_invocations_must_match_singleton_query_study(mutation):
    report = fixture_report()
    events = report["queries"][0]["candidate"]["frontier_invocations"]
    if mutation == "zero":
        events.clear()
    elif mutation == "two":
        events.append(deepcopy(events[0]))
    elif mutation == "reuse":
        events[0]["invocation_id"] = "q0-baseline"
    elif mutation == "query":
        events[0]["query_sha256"] = "a" * 64
    else:
        events[0]["requested_subset_signature"] = "a" * 64
    assert not evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())[
        "eligible"
    ]
