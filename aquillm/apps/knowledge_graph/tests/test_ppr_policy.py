import hashlib
import json
from dataclasses import replace

import pytest

from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind


def _signals(**changes):
    from apps.knowledge_graph.retrieval.ppr_policy import PPRPolicySignalsV1

    values = {
        "branch_kind": HybridBranchKind.DIRECT,
        "intent": "balanced",
        "seed_count": 1,
        "support_status": "supported",
        "cap_pressure": False,
        "outward_seed_mass": 1.0,
        "support_digest": "a" * 64,
    }
    values.update(changes)
    return PPRPolicySignalsV1(**values)


def test_relational_exploration_requires_support_and_connections():
    from apps.knowledge_graph.retrieval.ppr_policy import choose_ppr_restart

    signals = _signals(intent="relational", seed_count=2)
    assert choose_ppr_restart(signals, mode="adaptive").restart == 0.15
    assert choose_ppr_restart(
        replace(signals, support_status="unknown"), mode="adaptive"
    ).restart == 0.20
    assert choose_ppr_restart(
        replace(signals, cap_pressure=True), mode="adaptive"
    ).restart == 0.20
    assert choose_ppr_restart(
        replace(signals, intent="focused"), mode="adaptive"
    ).restart == 0.35


@pytest.mark.parametrize(
    ("query", "ambiguous", "expected"),
    (
        ("Who wrote this?", False, "focused"),
        ("Compare who wrote this and when", False, "balanced"),
        ("Who has a relationship between these entities?", False, "relational"),
        ("A title: who wrote this?", False, "balanced"),
        ("who wrote this?\nUse the supplied context.", False, "balanced"),
        ("who wrote this?", True, "balanced"),
        ("whenish information", False, "balanced"),
        ("¿Quién escribió esto?", False, "balanced"),
    ),
)
def test_classifier_is_conservative_and_uses_bounded_cues(query, ambiguous, expected):
    from apps.knowledge_graph.retrieval.ppr_policy import classify_ppr_intent

    assert classify_ppr_intent(query, context_ambiguous=ambiguous) == expected


def test_fixed_mode_preserves_baseline_and_adaptive_reason_codes_are_bounded():
    from apps.knowledge_graph.retrieval.ppr_policy import choose_ppr_restart

    fixed = choose_ppr_restart(_signals(intent="focused"), mode="fixed")
    assert (fixed.restart, fixed.iterations, fixed.reason) == (0.20, 8, "fixed_mode")
    insufficient = choose_ppr_restart(
        _signals(support_status="insufficient"), mode="adaptive"
    )
    assert insufficient.reason == "seed_support_insufficient"
    connections = choose_ppr_restart(
        _signals(intent="relational", seed_count=1), mode="adaptive"
    )
    assert (connections.restart, connections.reason) == (
        0.20,
        "insufficient_connections",
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"branch_kind": "direct"},
        {"intent": "broad"},
        {"seed_count": False},
        {"seed_count": 0},
        {"seed_count": 65},
        {"support_status": "maybe"},
        {"cap_pressure": 0},
        {"outward_seed_mass": False},
        {"outward_seed_mass": float("nan")},
        {"outward_seed_mass": float("inf")},
        {"outward_seed_mass": 1.01},
        {"support_digest": "g" * 64},
        {"support_digest": "a" * 63},
    ),
)
def test_signals_reject_nonexact_or_out_of_bounds_values(changes):
    with pytest.raises((TypeError, ValueError)):
        _signals(**changes)


@pytest.mark.parametrize("mode", ("", "ADAPTIVE", False))
def test_policy_rejects_unsupported_modes(mode):
    from apps.knowledge_graph.retrieval.ppr_policy import choose_ppr_restart

    with pytest.raises(ValueError):
        choose_ppr_restart(_signals(), mode=mode)


def test_policy_input_digest_is_stable_and_field_order_independent():
    from apps.knowledge_graph.retrieval.ppr_policy import ppr_policy_input_digest

    signals = _signals(intent="relational", seed_count=2, outward_seed_mass=0.5)
    expected_payload = {
        "support_digest": signals.support_digest,
        "outward_seed_mass": signals.outward_seed_mass.hex(),
        "cap_pressure": signals.cap_pressure,
        "support_status": signals.support_status,
        "seed_count": signals.seed_count,
        "intent": signals.intent,
        "branch_kind": signals.branch_kind.value,
    }
    expected = hashlib.sha256(
        json.dumps(
            expected_payload, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()

    assert ppr_policy_input_digest(signals) == expected
    assert ppr_policy_input_digest(signals) == ppr_policy_input_digest(signals)


def test_decision_rejects_values_outside_the_versioned_contract():
    from apps.knowledge_graph.retrieval.ppr_policy import PPRPolicyDecisionV1

    with pytest.raises(ValueError):
        PPRPolicyDecisionV1("ppr_restart_policy_v1", 0.50, 8, "balanced_intent")
    with pytest.raises(TypeError):
        PPRPolicyDecisionV1("ppr_restart_policy_v1", 0.20, True, "balanced_intent")
    with pytest.raises(ValueError):
        PPRPolicyDecisionV1("ppr_restart_policy_v1", 0.20, 8, "other")
