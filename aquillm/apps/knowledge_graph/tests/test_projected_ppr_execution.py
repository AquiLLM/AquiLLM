"""Adaptive execution keeps the projected topology contract authoritative."""

from dataclasses import replace
from hashlib import sha256

import pytest

from apps.knowledge_graph.retrieval import projected_ppr_execution as execution_module
from apps.knowledge_graph.retrieval.ppr import (
    canonical_algorithm_json,
)
from apps.knowledge_graph.retrieval.ppr_kernel import WeightedEdge, run_ppr_kernel
from apps.knowledge_graph.retrieval.ppr_policy import PPRPolicySignalsV1
from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
from apps.knowledge_graph.retrieval.projected_ppr_execution import (
    execute_adaptive_projected_ppr,
)
from apps.knowledge_graph.retrieval.projected_types import projected_snapshot_checksum
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
)
from apps.knowledge_graph.tests.projected_ppr_fixtures import key, projected_snapshot


def _fixture(*, edges=(("a", "b"),), config_changes=None):
    snapshot, config = projected_snapshot(edges=edges)
    config = replace(config, **({"ppr_iterations": 8} | (config_changes or {})))
    signature = sha256(
        b"ppr_projected_v1\0" + canonical_algorithm_json(config)
    ).hexdigest()
    snapshot = replace(
        snapshot,
        caps=replace(
            snapshot.caps,
            **{
                name: value
                for name, value in (config_changes or {}).items()
                if name in {"max_nodes", "max_edges"}
            },
        ),
        algorithm=replace(snapshot.algorithm, algorithm_signature=signature),
    )
    return snapshot, config


def _signals(**changes):
    values = {
        "branch_kind": HybridBranchKind.DIRECT,
        "intent": "balanced",
        "seed_count": 1,
        "support_status": "supported",
        "cap_pressure": False,
        "outward_seed_mass": 0.0,
        "support_digest": "a" * 64,
    }
    return PPRPolicySignalsV1(**(values | changes))


def _execute(snapshot, config, seeds, signals=None, deadline_check=lambda: None):
    return execute_adaptive_projected_ppr(
        snapshot=snapshot,
        seeds=seeds,
        base_config=config,
        signals=signals or _signals(seed_count=len(seeds)),
        expected_branch=HybridBranchKind.DIRECT,
        deadline_check=deadline_check,
    )


def test_two_node_recurrence_and_eight_step_execution() -> None:
    snapshot, config = _fixture()
    snapshot = replace(
        snapshot,
        identity_keys=(key("a"), key("b"))
        if key("a") < key("b")
        else (key("b"), key("a")),
        audit_rows=tuple(
            row
            for row in snapshot.audit_rows
            if getattr(row, "automatic_membership_key", None) != key("c")
        ),
    )
    seed = (ProjectedSeedV1(key("a"), 1.0),)
    first = run_ppr_kernel(
        nodes=(key("a"), key("b")),
        edges=(WeightedEdge(key("a"), key("b"), 1.0),),
        seeds={key("a"): 1.0},
        config=replace(config, ppr_iterations=1),
        order_key=lambda value: value,
    )
    assert dict(first.scores)[key("a")] == pytest.approx(0.2)
    assert dict(first.scores)[key("b")] == pytest.approx(0.8)

    result = _execute(snapshot, config, seed)
    a, b = 1.0, 0.0
    for _ in range(8):
        a, b = 0.2 + 0.8 * b, 0.8 * a
    assert result.decision.iterations == 8
    assert dict(result.ranking.scores) == {
        key("a"): pytest.approx(a),
        key("b"): pytest.approx(b),
    }


def test_balanced_adaptive_matches_legacy_trace_with_separate_identity() -> None:
    snapshot, config = _fixture()
    seeds = (ProjectedSeedV1(key("a"), 1.0),)
    original_checksum = projected_snapshot_checksum(snapshot)
    legacy = ppr_projected_v1(snapshot=snapshot, seeds=seeds, config=config)
    adaptive = _execute(snapshot, config, seeds)
    assert adaptive.ranking == legacy
    assert adaptive.decision.restart == 0.20
    assert adaptive.algorithm_signature != snapshot.algorithm.algorithm_signature
    assert adaptive.execution_signature != adaptive.algorithm_signature
    assert projected_snapshot_checksum(snapshot) == original_checksum


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("signature", "algorithm signature"),
        ("unauthorized_seed", "snapshot identities"),
        ("reordered_seeds", "sorted"),
        ("seed_count", "seed_count"),
        ("branch", "branch"),
    ],
)
def test_rejects_invalid_original_inputs_before_policy(change, message) -> None:
    snapshot, config = _fixture()
    seeds = (ProjectedSeedV1(key("a"), 1.0),)
    signals = _signals()
    if change == "signature":
        snapshot = replace(
            snapshot,
            algorithm=replace(snapshot.algorithm, algorithm_signature="f" * 64),
        )
    elif change == "unauthorized_seed":
        seeds = (ProjectedSeedV1(key("outsider"), 1.0),)
    elif change == "reordered_seeds":
        seeds = (ProjectedSeedV1(key("a"), 0.5), ProjectedSeedV1(key("b"), 0.5))
        signals = _signals(seed_count=2)
    elif change == "seed_count":
        signals = _signals(seed_count=2)
    else:
        signals = _signals(branch_kind=HybridBranchKind.EXTENDED)
    with pytest.raises((TypeError, ValueError), match=message):
        _execute(snapshot, config, seeds, signals)


def test_rejects_base_config_changes_even_when_caps_match() -> None:
    snapshot, config = _fixture()
    seeds = (ProjectedSeedV1(key("a"), 1.0),)
    with pytest.raises(ValueError, match="algorithm signature"):
        _execute(snapshot, replace(config, ppr_restart=0.35), seeds)
    snapshot, config = _fixture(config_changes={"ppr_iterations": 2})
    with pytest.raises(ValueError, match="eight"):
        _execute(snapshot, config, seeds)


def test_topology_overrides_claimed_outward_mass_and_combines_cap_pressure() -> None:
    snapshot, config = _fixture(edges=())
    seeds = (ProjectedSeedV1(key("a"), 1.0),)
    relational = _signals(intent="relational", outward_seed_mass=1.0)
    disconnected = _execute(snapshot, config, seeds, relational)
    assert disconnected.decision.restart == 0.20
    assert disconnected.decision.reason == "insufficient_connections"

    snapshot, config = _fixture(config_changes={"max_nodes": 3, "max_edges": 30})
    capped = _execute(snapshot, config, seeds, _signals(intent="focused"))
    assert capped.decision.reason == "cap_pressure"
    asserted = _execute(
        snapshot, config, seeds, _signals(intent="focused", cap_pressure=True)
    )
    assert asserted.decision.reason == "cap_pressure"


def test_relational_restart_uses_admitted_outward_seed_mass() -> None:
    snapshot, config = _fixture()
    seeds = tuple(
        sorted(
            (ProjectedSeedV1(key("a"), 0.5), ProjectedSeedV1(key("b"), 0.5)),
            key=lambda seed: seed.identity_key,
        )
    )
    result = _execute(
        snapshot,
        config,
        seeds,
        _signals(intent="relational", seed_count=2, outward_seed_mass=0.0),
    )
    assert result.decision.reason == "relational_supported"
    assert result.decision.restart == 0.15


def test_execution_identity_binds_effective_restart_and_absolute_seed_support() -> None:
    snapshot, config = _fixture()
    seeds = (ProjectedSeedV1(key("a"), 1.0),)
    balanced = _execute(snapshot, config, seeds)
    focused = _execute(snapshot, config, seeds, _signals(intent="focused"))
    assert focused.decision.restart == 0.35
    assert focused.algorithm_signature != balanced.algorithm_signature
    assert focused.execution_signature != balanced.execution_signature
    different_support = _execute(
        snapshot, config, seeds, _signals(support_digest="b" * 64)
    )
    assert different_support.ranking == balanced.ranking
    assert different_support.algorithm_signature == balanced.algorithm_signature
    assert different_support.policy_input_digest != balanced.policy_input_digest
    assert different_support.execution_signature != balanced.execution_signature


def test_kernel_runs_once_with_only_restart_changed(monkeypatch) -> None:
    snapshot, config = _fixture()
    seeds = (ProjectedSeedV1(key("a"), 1.0),)
    observed = []
    real_kernel = execution_module.run_ppr_kernel

    def run_and_capture(**kwargs):
        observed.append(kwargs["config"])
        return real_kernel(**kwargs)

    monkeypatch.setattr(execution_module, "run_ppr_kernel", run_and_capture)
    result = _execute(snapshot, config, seeds, _signals(intent="focused"))
    assert observed == [replace(config, ppr_restart=0.35)]
    assert result.decision.restart == 0.35


def test_execution_record_rejects_malformed_result_or_digest() -> None:
    snapshot, config = _fixture()
    result = _execute(snapshot, config, (ProjectedSeedV1(key("a"), 1.0),))
    with pytest.raises(TypeError, match="ranking"):
        replace(result, ranking="invalid")
    with pytest.raises(ValueError, match="execution_signature"):
        replace(result, execution_signature="not-a-digest")


def test_deadline_interrupts_without_mutating_original_snapshot() -> None:
    snapshot, config = _fixture()
    checksum = projected_snapshot_checksum(snapshot)
    seeds = (ProjectedSeedV1(key("a"), 1.0),)
    checks = 0

    def stop_during_kernel() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise TimeoutError("deadline")

    with pytest.raises(TimeoutError, match="deadline"):
        _execute(snapshot, config, seeds, deadline_check=stop_during_kernel)
    assert checks == 3
    assert projected_snapshot_checksum(snapshot) == checksum
