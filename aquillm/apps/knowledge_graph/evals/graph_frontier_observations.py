"""Request-global accounting originals. Missing V1/V2 capture stays unknown."""

from .graph_gate_observations import digest, require, sha

LIMITS = (
    "max_queries",
    "max_rows",
    "max_bytes",
    "max_identities",
    "max_edges",
    "max_depth",
)
COUNTERS = ("queries_used", "rows_used", "bytes_used", "identities_used", "edges_used")


def frontier_identity(value):
    return {k: value[k] for k in ("policy_signature", *LIMITS)}


def combined_graph_budget(controls, frontier):
    from .graph_effective_controls import graph_budget_digest

    return digest(
        dict(
            settings=graph_budget_digest(controls), frontier=frontier_identity(frontier)
        )
    )


def validate_invocations(row):
    events = row.get("frontier_invocations")
    require(
        type(events) is list and len(events) == 1, "graph_singleton_invocation_required"
    )
    event = events[0]
    require(
        type(event) is dict
        and set(event)
        == {
            "invocation_id",
            "query_sha256",
            "requested_subset_signature",
            "frontier_budget",
        },
        "graph_invocation_unknown",
    )
    require(
        type(event["invocation_id"]) is str and 0 < len(event["invocation_id"]) <= 128,
        "graph_invocation_unknown",
    )
    require(
        event["query_sha256"] == digest(row["inputs"]["question"])
        and sha(event["requested_subset_signature"])
        and event["frontier_budget"] == row["frontier_budget"],
        "graph_invocation_binding_mismatch",
    )


def validate_frontier(value):
    require(
        type(value) is dict
        and set(value)
        == {"policy_signature", *LIMITS, *COUNTERS, "accounting_complete", "uncertain"},
        "frontier_accounting_unknown",
    )
    require(sha(value["policy_signature"]), "frontier_policy_unknown")
    for key in (*LIMITS, *COUNTERS):
        require(
            type(value[key]) is int and 0 <= value[key] <= 2**53 - 1,
            "frontier_accounting_invalid",
        )
    require(all(value[k] > 0 for k in LIMITS), "frontier_limits_invalid")
    require(
        value["max_queries"] <= 64
        and value["max_rows"] <= 5000
        and value["max_bytes"] <= 1048576
        and value["max_identities"] <= 200
        and value["max_edges"] <= 1000
        and value["max_depth"] <= 2,
        "frontier_limits_invalid",
    )
    require(
        type(value["accounting_complete"]) is bool and type(value["uncertain"]) is bool,
        "frontier_accounting_unknown",
    )
    # Originals are retained unchanged, including overshoot and uncertain work.
    require(not value["uncertain"], "frontier_accounting_uncertain")
    require(value["accounting_complete"], "frontier_accounting_incomplete")
    require(
        all(value[counter] <= value[limit] for counter, limit in zip(COUNTERS, LIMITS)),
        "frontier_budget_exceeded",
    )
