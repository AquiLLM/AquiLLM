"""Original counters survive retrieval sealing and delayed synthesis."""

import asyncio

import pytest

from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def test_sealed_synthesis_survives_expiry_but_not_cancellation():
    from lib.retrieval.synthesis_budget import seal_synthesis

    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    assert budget.reserve_action("initial")
    lease = seal_synthesis(
        budget, "exact payload", calls=3, output_tokens=2000, timeout_seconds=60
    )
    now[0] = 16
    assert not budget.can_publish()
    assert not budget.reserve_action("late")
    assert lease.reserve_text(100, kind="tokenized")
    assert lease.can_publish()
    assert budget.text_used["tokenized"] == 100
    assert budget.actions_used == 1
    budget.close("cancelled")
    with pytest.raises(asyncio.CancelledError):
        lease.check_active()


def test_expired_unsealed_evidence_cannot_start_synthesis():
    from lib.retrieval.synthesis_budget import seal_synthesis

    budget = TurnBudget(TurnLimits(retrieval_ms=0, final_scoring_ms=0))
    with pytest.raises(ValueError):
        seal_synthesis(
            budget, "payload", calls=3, output_tokens=2000, timeout_seconds=60
        )
