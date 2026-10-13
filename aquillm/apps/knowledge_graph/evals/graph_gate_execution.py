# Exact graph execution observations, separate from held-out labels.
from .graph_frontier_observations import validate_frontier, validate_invocations
from .graph_gate_observations import ids, number, require, sha


def validate_arm(row, query_id, expected_binding):
    fields = {
        "binding",
        "inputs",
        "ranked",
        "repeated_ranked",
        "abstained",
        "coverage",
        "exhaustive",
        "authorization_violations",
        "citation_violations",
        "completion",
        "cache_state",
        "provider_cache",
        "latency_ms",
        "latency_budget_ms",
        "frontier_budget",
        "frontier_invocations",
        "request_algorithm_signature",
    }
    require(type(row) is dict and set(row) == fields, "invalid_execution_observation")
    require(row["binding"] == expected_binding, "comparison_binding_mismatch")
    validate_frontier(row["frontier_budget"])
    inputs = row["inputs"]
    require(
        type(inputs) is dict
        and set(inputs) == {"query_id", "question", "scope_sha256"},
        "execution_inputs_contain_labels_or_unknown_fields",
    )
    require(
        inputs["query_id"] == query_id
        and type(inputs["question"]) is str
        and inputs["question"]
        and sha(inputs["scope_sha256"]),
        "invalid_execution_inputs",
    )
    validate_invocations(row)
    require(
        sha(row["request_algorithm_signature"])
        and row["request_algorithm_signature"] != row["binding"]["algorithm_signature"],
        "graph_request_signature_unknown",
    )
    ids(row["ranked"])
    ids(row["repeated_ranked"])
    require(
        type(row["abstained"]) is bool and type(row["exhaustive"]) is bool,
        "unknown_abstention_or_coverage",
    )
    require(not row["abstained"] or not row["ranked"], "contradictory_abstention")
    require(row["exhaustive"] is False, "bounded_graph_exhaustive_claim")
    require(
        row["coverage"] in ("complete", "partial", "unavailable"), "coverage_unknown"
    )
    require(
        row["cache_state"] in ("cold", "warm")
        and row["provider_cache"] in ("cold", "warm", "unknown"),
        "cache_observation_unknown",
    )
    require(type(row["completion"]) is str, "completion_unknown")
    for name in ("authorization_violations", "citation_violations"):
        require(type(row[name]) is int and row[name] >= 0, "safety_unknown")
    number(row["latency_ms"])
    require(0 < number(row["latency_budget_ms"]) <= 5000, "latency_budget_unknown")
