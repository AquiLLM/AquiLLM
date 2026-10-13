"""Preregistered necessary graph adoption predicates, not activation authority."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .graph_gate_metrics import identity_metrics, retrieval_metrics
from .graph_gate_observations import digest, validate_report


@dataclass(frozen=True, slots=True)
class GraphGatePolicyV2:
    min_automatic_precision: float = 0.995
    min_automatic_precision_lower_95: float = 0.99
    min_automatic_pairs: int = 600
    min_query_pairs: int = 150
    min_pairs_per_stratum: int = 50
    max_metric_decline: float = 0.0


def evaluate_graph_gates_v2(*, report, policy):
    reasons, identity, retrieval = [], {}, {}
    # Policy is frozen before collection; candidates cannot relax it with flags.
    expected = asdict(GraphGatePolicyV2())
    if type(policy) is not GraphGatePolicyV2 or any(
        type(value) is not type(expected[key]) or value != expected[key]
        for key, value in asdict(policy).items()
    ):
        reasons.append("unrecognized_graph_policy")
    elif (
        not isinstance(report, Mapping) or report.get("schema") != "graph-evaluation-v2"
    ):
        reasons.append("legacy_report_ineligible")
    else:
        if report.get("backend") != "live":
            reasons.append("synthetic_evidence")
        if report.get("policy_sha256") != digest(asdict(policy)):
            reasons.append("graph_policy_digest_mismatch")
        try:
            validate_report(report)
            identity = identity_metrics(report["decisions"], policy, reasons)
            retrieval = retrieval_metrics(
                report["queries"], policy, report["intended_improvement"], reasons
            )
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            reasons.append("invalid_or_missing_observation:" + str(error))
    return dict(
        gate_version="graph-quality-v2",
        eligible=not reasons,
        blocking_reasons=sorted(set(reasons)),
        identity=identity,
        retrieval=retrieval,
        policy=asdict(GraphGatePolicyV2()),
    )
