"""One statistical question, four independently paired execution modes."""

from .graph_frontier_observations import frontier_identity
from .graph_gate_execution import validate_arm
from .graph_gate_observations import require
from .graph_study_controls import validate_execution_controls

MODES = ("baseline", "selection", "preservation", "combined")


def validate_modes(query, report, invocations_seen):
    modes = query["executions"]
    require(
        type(modes) is dict and set(modes) == set(MODES),
        "graph_modes_missing_or_unknown",
    )
    inputs = None
    for pair in modes.values():
        require(
            type(pair) is dict and set(pair) == {"baseline", "candidate"},
            "graph_mode_pair_unknown",
        )
        for arm in ("baseline", "candidate"):
            observation = pair[arm]
            validate_arm(
                observation,
                query["query_id"],
                report["baseline_binding" if arm == "baseline" else "binding"],
            )
            validate_execution_controls(report, arm, observation)
            invocation = observation["frontier_invocations"][0]["invocation_id"]
            require(invocation not in invocations_seen, "graph_invocation_reused")
            invocations_seen.add(invocation)
            if inputs is None:
                inputs = observation["inputs"]
            require(observation["inputs"] == inputs, "unmatched_query_inputs")
        require(
            frontier_identity(pair["baseline"]["frontier_budget"])
            == frontier_identity(pair["candidate"]["frontier_budget"]),
            "unmatched_frontier_budget",
        )
        require(
            pair["baseline"]["frontier_invocations"][0]["requested_subset_signature"]
            == pair["candidate"]["frontier_invocations"][0][
                "requested_subset_signature"
            ],
            "unmatched_invocation_subset",
        )
        require(
            pair["baseline"]["cache_state"] == pair["candidate"]["cache_state"]
            and pair["baseline"]["provider_cache"]
            == pair["candidate"]["provider_cache"],
            "unmatched_cache_state",
        )
    return inputs
