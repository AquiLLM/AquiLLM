"""Versioned adaptive PPR execution over an unchanged projected snapshot."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from math import fsum

from .ppr import PPRAlgorithmConfig, canonical_algorithm_json
from .ppr_kernel import run_ppr_kernel
from .ppr_policy import (
    PPRPolicyDecisionV1,
    PPRPolicySignalsV1,
    choose_ppr_restart,
    ppr_policy_input_digest,
)
from .projected_ppr import (
    prepare_projected_ppr_inputs,
)
from .projected_ppr_result import ProjectedPPRResultV1, _trace_bytes
from .projected_types import (
    ProjectedAuthorizedGraphSnapshotV1,
    projected_snapshot_checksum,
)
from .topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
    projected_seed_checksum,
)


@dataclass(frozen=True, slots=True)
class ProjectedPPRExecutionV1:
    ranking: ProjectedPPRResultV1
    decision: PPRPolicyDecisionV1
    algorithm_signature: str
    execution_signature: str
    policy_input_digest: str

    def __post_init__(self) -> None:
        if type(self.ranking) is not ProjectedPPRResultV1:
            raise TypeError("ranking must be an exact ProjectedPPRResultV1")
        if type(self.decision) is not PPRPolicyDecisionV1:
            raise TypeError("decision must be an exact PPRPolicyDecisionV1")
        for name in (
            "algorithm_signature",
            "execution_signature",
            "policy_input_digest",
        ):
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError(f"{name} must be an exact string")
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _digest(payload: dict[str, object]) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def prepare_projected_ppr_policy(
    *,
    snapshot: ProjectedAuthorizedGraphSnapshotV1,
    seeds: tuple[ProjectedSeedV1, ...],
    base_config: PPRAlgorithmConfig,
    signals: PPRPolicySignalsV1,
    expected_branch: HybridBranchKind,
    deadline_check: Callable[[], None],
) -> tuple:
    """Admit topology and derive one policy proposal without running PPR."""
    if not callable(deadline_check):
        raise TypeError("deadline_check must be callable")
    deadline_check()
    edges = prepare_projected_ppr_inputs(
        snapshot=snapshot, seeds=seeds, config=base_config
    )
    if base_config.ppr_iterations != 8:
        raise ValueError("adaptive PPR requires eight iterations")
    if type(signals) is not PPRPolicySignalsV1:
        raise TypeError("signals must be an exact PPRPolicySignalsV1")
    if type(expected_branch) is not HybridBranchKind:
        raise TypeError("expected_branch must be an exact HybridBranchKind")
    if signals.branch_kind is not expected_branch:
        raise ValueError("signal branch does not match expected branch")
    if signals.seed_count != len(seeds):
        raise ValueError("signal seed_count does not match admitted seeds")

    outward_sources = {
        edge.source
        for edge in edges
        if edge.weight > 0.0 and edge.source != edge.target
    }
    outward_seed_mass = min(
        1.0, fsum(seed.mass for seed in seeds if seed.identity_key in outward_sources)
    )
    cap_pressure = (
        signals.cap_pressure
        or len(snapshot.identity_keys) >= base_config.max_nodes
        or len(edges) >= base_config.max_edges
    )
    admitted_signals = replace(
        signals, outward_seed_mass=outward_seed_mass, cap_pressure=cap_pressure
    )
    decision = choose_ppr_restart(admitted_signals, mode="adaptive")
    return edges, admitted_signals, decision


def execute_adaptive_projected_ppr(
    *,
    snapshot: ProjectedAuthorizedGraphSnapshotV1,
    seeds: tuple[ProjectedSeedV1, ...],
    base_config: PPRAlgorithmConfig,
    signals: PPRPolicySignalsV1,
    expected_branch: HybridBranchKind,
    deadline_check: Callable[[], None],
) -> ProjectedPPRExecutionV1:
    """Validate original topology, decide once, then run exactly one PPR kernel."""
    edges, admitted_signals, decision = prepare_projected_ppr_policy(
        snapshot=snapshot,
        seeds=seeds,
        base_config=base_config,
        signals=signals,
        expected_branch=expected_branch,
        deadline_check=deadline_check,
    )
    effective_config = replace(base_config, ppr_restart=decision.restart)
    policy_digest = ppr_policy_input_digest(admitted_signals)
    algorithm_signature = _digest(
        {
            "domain": "ppr_projected_execution_v1",
            "effective_config": json.loads(canonical_algorithm_json(effective_config)),
            "policy_version": decision.policy_version,
        }
    )
    execution_signature = _digest(
        {
            "domain": "ppr_projected_execution_v1",
            "algorithm_signature": algorithm_signature,
            "snapshot_checksum": projected_snapshot_checksum(snapshot),
            "seed_checksum": projected_seed_checksum(seeds),
            "policy_input_digest": policy_digest,
        }
    )
    kernel = run_ppr_kernel(
        nodes=snapshot.identity_keys,
        edges=edges,
        seeds={seed.identity_key: seed.mass for seed in seeds},
        config=effective_config,
        order_key=lambda key: key,
        deadline_check=deadline_check,
    )
    scores = tuple(kernel.scores)
    score_map = dict(scores)
    ranked = tuple(sorted(score_map, key=lambda key: (-score_map[key], key)))
    return ProjectedPPRExecutionV1(
        ProjectedPPRResultV1(scores, ranked, _trace_bytes(scores, ranked)),
        decision,
        algorithm_signature,
        execution_signature,
        policy_digest,
    )


__all__ = [
    "ProjectedPPRExecutionV1",
    "execute_adaptive_projected_ppr",
    "prepare_projected_ppr_policy",
]
