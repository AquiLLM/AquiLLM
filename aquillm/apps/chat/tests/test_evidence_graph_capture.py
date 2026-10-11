"""Original G3 events must survive evaluation without inventing graph authority."""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic

import pytest

from apps.chat.evals.evidence_graph_capture import graph_lane_observations
from apps.chat.evals.evidence_observation_json import normalize_observation
from apps.chat.evals.evidence_quality_trace import Trace
from apps.knowledge_graph.retrieval.graph_frontier_observation_v2 import (
    capture_frontier_budget_v2,
    publish_frontier_budget_v2,
)
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetV2,
)
from lib.evidence_observation import observe
from lib.retrieval import TurnBudget, TurnLimits


def publish_actual(trace, *, overshoot=False, signature="a" * 64):
    budget = FrontierBudgetV2(
        turn_budget=TurnBudget(TurnLimits()),
        deadline=monotonic() + 2,
        max_identities=10,
        max_edges=20,
        max_source_rows=30,
        max_queries=3,
        max_wire_bytes=1024,
        clock=monotonic,
    )
    assert budget.reserve_query()
    budget.charge_page(scanned_rows=31 if overshoot else 2, wire_bytes=80)
    original = capture_frontier_budget_v2(budget, max_depth=1)
    with observe(trace.sink):
        event = publish_frontier_budget_v2(
            original,
            query="public question",
            requested_document_ids=[],
            request_algorithm_signature=signature,
        )
    return asdict(event)


def test_actual_g3_producer_to_trace_retains_envelope_and_exact_invocation():
    trace = Trace({})
    trace.final["lane_observations"] = {"retrieval": {"status": "observed"}}
    original = publish_actual(trace)
    before = deepcopy(trace.events)
    lanes = trace.result("")["lane_observations"]
    graph = lanes["graph"]
    assert lanes["retrieval"] == {"status": "observed"}
    assert graph["frontier_invocations"] == [original]
    assert graph["frontier_events"] == before
    assert trace.events == before
    assert graph["frontier_capture_status"] == "singleton"
    assert graph["frontier_budget"] == original["frontier_budget"]
    assert (
        graph["request_algorithm_signature"] == original["request_algorithm_signature"]
    )
    assert graph["requested_subset_signature"] == original["requested_subset_signature"]
    assert graph["status"] == "unknown"
    assert "source_generation_sha256" not in graph
    assert "runtime_sha256" not in graph
    graph["frontier_invocations"][0]["frontier_budget"]["rows_used"] = 999
    assert trace.events == before


@pytest.mark.parametrize(
    "fault",
    [
        "zero",
        "multiple",
        "extra",
        "missing",
        "time",
        "infinite_time",
        "nan_time",
        "overshoot",
        "signature",
        "uncertain",
    ],
)
def test_missing_or_invalid_capture_retains_all_originals_and_blocks(fault):
    trace = Trace({})
    if fault != "zero":
        publish_actual(
            trace,
            overshoot=fault == "overshoot",
            signature=None if fault == "signature" else "a" * 64,
        )
    if fault == "multiple":
        publish_actual(trace)
    if fault == "extra":
        trace.events[0]["unrecognized"] = "retained"
    if fault == "missing":
        del trace.events[0]["at"]
    if fault == "time":
        trace.events[0]["at"] = -1.0
    if fault == "infinite_time":
        trace.events[0]["at"] = float("inf")
    if fault == "nan_time":
        trace.events[0]["at"] = float("nan")
    if fault == "uncertain":
        trace.events[0]["frontier_budget"]["uncertain"] = True
        trace.events[0]["frontier_budget"]["accounting_complete"] = False
    before = deepcopy(trace.events)
    raw = graph_lane_observations(trace.events)["graph"]
    assert raw["frontier_events"] == before
    result = trace.result("")
    graph = result["lane_observations"]["graph"]
    assert graph["status"] == "unknown"
    assert graph["frontier_capture_status"] == "inconclusive"
    assert (
        len(graph["frontier_events"])
        == len(graph["frontier_invocations"])
        == len(before)
    )
    assert (
        graph["frontier_events"] == normalize_observation({"events": before})["events"]
    )
    assert graph["frontier_invocations"] == [
        {k: v for k, v in event.items() if k not in ("event", "at")} for event in before
    ]
    assert "frontier_budget" not in graph
    assert "request_algorithm_signature" not in graph
    assert trace.events == before
    if fault in ("infinite_time", "nan_time"):
        assert "graph_transport_observation_unknown" in raw["frontier_capture_reasons"]
        assert result["observation_normalization_failed"] is True
        assert result["provenance_complete"] is False
        assert result["dispatch_accounting_complete"] is False
