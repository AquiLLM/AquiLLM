"""Retain graph producer originals without creating study or runtime authority."""

from copy import deepcopy
from math import isfinite

from apps.knowledge_graph.evals.graph_frontier_observations import validate_frontier
from apps.knowledge_graph.evals.graph_gate_observations import sha

_FIELDS = {
    "invocation_id",
    "query_sha256",
    "requested_subset_signature",
    "frontier_budget",
    "request_algorithm_signature",
}


def graph_lane_observations(events, existing=None):
    """Transport envelopes and five-field G3 payloads retain one-to-one order.

    A valid singleton establishes only capture completeness. Source-generation,
    runtime, current execution and independent study identities remain separate
    requirements of the original G6/A3 validators.
    """
    lanes = deepcopy(existing) if isinstance(existing, dict) else {}
    previous = lanes.get("graph")
    graph = deepcopy(previous) if isinstance(previous, dict) else {}
    originals = deepcopy(
        [event for event in events if event.get("event") == "graph_frontier_budget"]
    )
    invocations = [
        {key: value for key, value in event.items() if key not in ("event", "at")}
        for event in originals
    ]
    # Remove stale projections even if a previous capture was a singleton.
    for key in (*_FIELDS, "frontier_capture_reasons"):
        graph.pop(key, None)
    reasons = []
    if len(invocations) != 1:
        reasons.append("graph_singleton_invocation_required")
    else:
        event, original = invocations[0], originals[0]
        at = original.get("at")
        if type(at) not in (int, float) or not isfinite(at) or at < 0:
            reasons.append("graph_transport_observation_unknown")
        if set(event) != _FIELDS or not (
            type(event.get("invocation_id")) is str
            and 0 < len(event["invocation_id"]) <= 128
            and all(
                sha(event.get(key))
                for key in (
                    "query_sha256",
                    "requested_subset_signature",
                    "request_algorithm_signature",
                )
            )
        ):
            reasons.append("graph_invocation_unknown")
        try:
            validate_frontier(event.get("frontier_budget"))
        except (ValueError, TypeError, KeyError):
            reasons.append("graph_frontier_accounting_invalid")
        if not reasons:
            graph.update(deepcopy(event))
    graph.setdefault("status", "unknown")
    if reasons:
        graph["status"] = "unknown"
    graph.update(
        frontier_events=originals,
        frontier_invocations=invocations,
        frontier_capture_status="inconclusive" if reasons else "singleton",
        frontier_capture_reasons=reasons,
    )
    lanes["graph"] = graph
    return lanes
