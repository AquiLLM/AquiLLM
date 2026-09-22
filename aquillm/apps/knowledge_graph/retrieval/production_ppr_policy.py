"""One-run serving dispatch for fixed, shadow, and adaptive projected PPR."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic

import structlog

from .ppr import PPRAlgorithmConfig
from .ppr_policy import PPRPolicySignalsV1
from .ppr_seed_support import PreparedPPRSeedsV1
from .projected_ppr import ProjectedPPRResultV1, ppr_projected_v1
from .projected_ppr_execution import (
    execute_adaptive_projected_ppr,
    prepare_projected_ppr_policy,
)
from .projected_types import ProjectedAuthorizedGraphSnapshotV1
from .topology.contracts import HybridBranchKind, ProjectedSeedV1

_LOG = structlog.stdlib.get_logger(__name__)


def rank_projected_for_mode(
    *,
    snapshot: ProjectedAuthorizedGraphSnapshotV1,
    seeds: tuple[ProjectedSeedV1, ...],
    config: PPRAlgorithmConfig,
    prepared: PreparedPPRSeedsV1 | None,
    mode: str,
    branch: HybridBranchKind,
    deadline_check: Callable[[], None],
) -> tuple[ProjectedPPRResultV1, str | None]:
    """Keep fixed/shadow on legacy identity and adaptive on one new execution."""
    if type(mode) is not str or mode not in ("fixed", "shadow", "adaptive"):
        raise ValueError("unsupported PPR restart mode")
    if type(branch) is not HybridBranchKind:
        raise TypeError("branch must be an exact HybridBranchKind")
    started = monotonic()
    if mode == "fixed":
        ranking = ppr_projected_v1(snapshot=snapshot, seeds=seeds, config=config)
        signature = None
        proposed = effective = config.ppr_restart
        reason = "fixed_mode"
    else:
        if type(prepared) is not PreparedPPRSeedsV1 or prepared.seeds != seeds:
            raise ValueError("policy seeds do not match admitted seeds")
        signals = PPRPolicySignalsV1(
            branch_kind=branch,
            intent=prepared.intent,
            seed_count=len(seeds),
            support_status=prepared.support.status,
            cap_pressure=prepared.support.cap_pressure,
            outward_seed_mass=0.0,
            support_digest=prepared.support.digest,
        )
        if mode == "shadow":
            _edges, _admitted, decision = prepare_projected_ppr_policy(
                snapshot=snapshot,
                seeds=seeds,
                base_config=config,
                signals=signals,
                expected_branch=branch,
                deadline_check=deadline_check,
            )
            deadline_check()
            ranking = ppr_projected_v1(snapshot=snapshot, seeds=seeds, config=config)
            deadline_check()
            signature = None
            proposed, effective, reason = (
                decision.restart,
                config.ppr_restart,
                decision.reason,
            )
        else:
            execution = execute_adaptive_projected_ppr(
                snapshot=snapshot,
                seeds=seeds,
                base_config=config,
                signals=signals,
                expected_branch=branch,
                deadline_check=deadline_check,
            )
            ranking, signature = execution.ranking, execution.algorithm_signature
            proposed = effective = execution.decision.restart
            reason = execution.decision.reason
    _LOG.info(
        "obs.rag.ppr_restart_decision",
        branch_kind=branch.value,
        mode=mode,
        proposed_restart=proposed,
        effective_restart=effective,
        policy_version="ppr_restart_policy_v1",
        reason=reason,
        elapsed_ms=max(0, int((monotonic() - started) * 1000)),
    )
    return ranking, signature


__all__ = ["rank_projected_for_mode"]
