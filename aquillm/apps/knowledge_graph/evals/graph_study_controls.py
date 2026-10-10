"""Both graph arms derive identities from originals; declarations cannot waive caps."""

from .graph_effective_controls import validate_controls
from .graph_frontier_observations import combined_graph_budget
from .graph_gate_observations import digest, require

TREATMENT_PATHS = frozenset(
    {
        "identity_policy",
        "remediation_mode",
        "algorithm_config.ppr_restart",
        "algorithm_config.rrf_k",
        "runtime_observations.ppr_restart_mode",
        "runtime_observations.graph_direct_enabled",
        "runtime_observations.graph_extended_enabled",
    }
)


def meaningful(control):
    result = {}
    for key, value in control.items():
        if key in ("algorithm_signature", "runtime_sha256", "resolver_versions"):
            continue
        if type(value) is dict:
            result.update({key + "." + k: v for k, v in value.items()})
        else:
            result[key] = value
    return result


def validate_study_controls(report):
    controls = report["controls"]
    require(
        type(controls) is dict and set(controls) == {"baseline", "candidate"},
        "graph_study_controls_unknown",
    )
    for arm in ("baseline", "candidate"):
        control = controls[arm]
        validate_controls(control, observed_only=True)
        bound = report["baseline_binding" if arm == "baseline" else "binding"]
        require(
            bound["runtime"] == control["runtime_sha256"]
            and bound["algorithm_signature"] == control["algorithm_signature"],
            "graph_original_control_binding_mismatch",
        )
    before, after = (meaningful(controls[a]) for a in ("baseline", "candidate"))
    changed = {
        k: {"baseline": before[k], "candidate": after[k]}
        for k in before
        if before[k] != after[k]
    }
    canonical = "algorithm_config.canonical_resolver_version"
    if canonical in changed and "identity_policy" in changed:
        require(
            all(
                controls[a]["algorithm_config"]["canonical_resolver_version"]
                == controls[a]["resolver_versions"]["canonical"]
                for a in ("baseline", "candidate")
            ),
            "graph_resolver_treatment_mismatch",
        )
        del changed[canonical]  # Derived from the separately declared identity policy.
    require(set(changed) <= TREATMENT_PATHS, "graph_non_treatment_control_drift")
    require(
        type(report["treatment"]) is dict
        and digest(report["treatment"]) == digest(changed),
        "graph_treatment_declaration_mismatch",
    )


def validate_execution_controls(report, arm, execution):
    require(
        execution["binding"]["graph_budget"]
        == combined_graph_budget(report["controls"][arm], execution["frontier_budget"]),
        "graph_original_budget_binding_mismatch",
    )
    runtime = report["controls"][arm]["runtime_observations"]
    for branch in ("direct", "extended"):
        if not runtime[f"graph_{branch}_enabled"]:
            continue
        for limit, suffix in (
            ("max_identities", "nodes"),
            ("max_edges", "edges"),
            ("max_depth", "depth"),
        ):
            require(
                execution["frontier_budget"][limit]
                <= runtime[f"graph_{branch}_max_{suffix}"],
                "graph_frontier_exceeds_effective_caps",
            )
        require(
            len(execution["ranked"]) <= runtime[f"graph_{branch}_max_candidates"],
            "graph_results_exceed_effective_cap",
        )
