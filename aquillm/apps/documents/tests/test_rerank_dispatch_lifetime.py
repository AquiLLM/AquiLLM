"""Actual HTTP cannot begin from a stale admitted rerank worker allowance."""

import pytest

from .rerank_dispatch_test_support import (
    assert_original_lease_finished,
    paused_http_score,
)


KINDS = ("score_batch_text_pairs", "score_single_text_pair")
PHASES = ("acquisition", "final")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("transition", ("expired", "closed", "failure_latch"))
def test_admitted_worker_rechecks_original_lifetime_before_http(
    monkeypatch, kind, phase, transition
):
    with paused_http_score(monkeypatch, kind, phase) as case:
        if transition == "expired":
            case.clock[0] = 15.001 if phase == "acquisition" else 3.001
            assert case.budget.scoring_remaining_ms(phase) == 0
        elif transition == "closed":
            case.budget.close("cancelled")
            assert not case.budget.can_publish()
        else:
            case.state.fail()
            assert case.budget.can_publish()
            assert not case.state.can_admit()
        assert case.finish() is None
        assert_original_lease_finished(case, phase)
        assert case.dispatches == [], "closed work still called the HTTP transport"
        assert case.received == [], "closed work reached the real loopback provider"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("delayed", (False, True))
def test_live_worker_uses_remaining_original_allowance(monkeypatch, kind, phase, delayed):
    with paused_http_score(monkeypatch, kind, phase) as case:
        if delayed:
            case.clock[0] = 12.5 if phase == "acquisition" else 1.5
        expected = min(10, case.budget.scoring_remaining_ms(phase) / 1000)
        assert expected > 0 and case.state.can_admit()
        assert case.finish() == (0.0,)
        assert_original_lease_finished(case, phase)
        assert len(case.received) == 1
        payload = case.received[0]
        assert payload["model"] == "synthetic"
        assert payload["text_1"] == "question"
        assert payload["text_2"] == (
            ["original source"] if kind == "score_batch_text_pairs" else "original source"
        )
        assert len(case.dispatches) == 1
        timeout, remaining_ms = case.dispatches[0]
        assert remaining_ms > 0
        assert timeout == pytest.approx(expected)
        assert 0 < timeout <= remaining_ms / 1000
