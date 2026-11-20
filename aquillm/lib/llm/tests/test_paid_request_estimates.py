"""Paid tokenization is reusable work, never a publication authorization."""

import asyncio
from dataclasses import replace

import pytest

from lib.llm import evidence_guard as guard
from lib.llm.utils import evidence_tokens
from lib.retrieval.synthesis_budget import seal_synthesis
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


PAYLOAD = {"messages": [{"role": "user", "content": "Selected evidence."}]}


def budget(clock=None, **limits):
    return TurnBudget(TurnLimits(**limits), clock=clock or (lambda: 0.0))


def prepay(payload, owner, *, pin=False):
    # Before the feature exists this exercises the real original charging path.
    function = getattr(guard, "prepay_request_tokens", None)
    if function is None:
        return guard.estimate_request_tokens(payload, turn_budget=owner)
    return function(payload, turn_budget=owner, pin=pin)


def lease(owner):
    return seal_synthesis(owner, "Selected evidence.", calls=2,
        output_tokens=100, timeout_seconds=10, authority=object())


def test_prepaid_exact_request_survives_no_remaining_tokenization_capacity():
    owner = budget()
    expected = prepay(PAYLOAD, owner, pin=True)
    paid = owner.text_used["tokenized"]
    assert owner.reserve_text(owner.limits.tokenized_codepoints - paid, kind="tokenized")
    assert guard.estimate_request_tokens(PAYLOAD, turn_budget=lease(owner)) == expected
    assert owner.text_used["tokenized"] == owner.limits.tokenized_codepoints


def test_unpaid_requests_keep_charging_each_actual_tokenization():
    owner = budget()
    guard.estimate_request_tokens(PAYLOAD, turn_budget=owner)
    first = owner.text_used["tokenized"]
    guard.estimate_request_tokens(PAYLOAD, turn_budget=owner)
    assert owner.text_used["tokenized"] == 2 * first


@pytest.mark.parametrize("change", ["text", "stream", "model", "format"])
def test_changed_provider_request_cannot_spend_an_old_paid_estimate(change):
    owner = budget()
    prepay(PAYLOAD, owner, pin=True)
    before = owner.text_used["tokenized"]
    changed = {"messages": [dict(PAYLOAD["messages"][0])]}
    if change == "text":
        changed["messages"][0]["content"] += " Another claim."
    elif change == "stream":
        changed.update(stream=True, stream_options={"include_usage": True})
    elif change == "model":
        changed["model"] = "another-model"
    else:
        changed["response_format"] = {"type": "json_object"}
    guard.estimate_request_tokens(changed, turn_budget=owner)
    assert owner.text_used["tokenized"] > before


def test_paid_estimate_cannot_move_to_another_ledger():
    first, second = budget(), budget()
    prepay(PAYLOAD, first, pin=True)
    if hasattr(first, "_paid_request_estimates"):
        second._paid_request_estimates = first._paid_request_estimates
    guard.estimate_request_tokens(PAYLOAD, turn_budget=second)
    assert second.text_used["tokenized"] > 0


def test_native_registered_lease_is_required_for_a_paid_hit():
    owner = budget()
    prepay(PAYLOAD, owner, pin=True)
    original = lease(owner)
    forged = replace(original, authority=object())
    with pytest.raises((ValueError, guard.ContextLimited), match="synthesis allowance"):
        guard.estimate_request_tokens(PAYLOAD, turn_budget=forged)


@pytest.mark.parametrize("expired", ["retrieval", "synthesis", "closed", "cancelled"])
def test_paid_hit_cannot_bypass_deadline_or_cancellation(expired):
    now = [0.0]
    owner = budget(clock=lambda: now[0])
    prepay(PAYLOAD, owner, pin=True)
    active = lease(owner) if expired in {"synthesis", "cancelled"} else owner
    if expired == "closed":
        owner.close("closed")
    elif expired == "cancelled":
        owner.close("cancelled")
    else:
        now[0] = 1000.0
    with pytest.raises((ValueError, guard.ContextLimited, asyncio.CancelledError)):
        guard.estimate_request_tokens(PAYLOAD, turn_budget=active)


@pytest.mark.parametrize("change", ["evidence", "context", "output"])
def test_paid_hit_still_checks_current_evidence_and_context(change):
    owner = budget()
    prepay(PAYLOAD, owner, pin=True)
    protection = guard.EvidenceProtection("Selected evidence.", 10000, 10, 0,
        turn_budget=owner, synthesis_lease=lease(owner))
    if change == "evidence":
        protection.payload = "Different authorized evidence."
    elif change == "context":
        protection.model_context = 1
    else:
        protection.output_reserve = 10000
    with guard.protect_evidence(protection), pytest.raises(guard.ContextLimited):
        guard.validate_request(PAYLOAD, output_reserve=10)


def test_policy_change_invalidates_paid_tokenization(monkeypatch):
    owner = budget()
    prepay(PAYLOAD, owner, pin=True)
    before = owner.text_used["tokenized"]
    original = evidence_tokens._ENCODING

    class DifferentPolicy:
        def encode(self, *args, **kwargs):
            return original.encode(*args, **kwargs)

    monkeypatch.setattr(evidence_tokens, "_ENCODING", DifferentPolicy())
    guard.estimate_request_tokens(PAYLOAD, turn_budget=owner)
    assert owner.text_used["tokenized"] > before


def test_cache_keeps_ordinary_and_only_latest_combined_without_refunding():
    owner = budget()
    ordinary = prepay(PAYLOAD, owner, pin=True)
    one, two = {"content": "Combined one."}, {"content": "Combined two."}
    prepay(one, owner)
    before = owner.text_used["tokenized"]
    combined = prepay(two, owner)
    paid = owner.text_used["tokenized"]
    assert paid > before
    assert guard.estimate_request_tokens(PAYLOAD, turn_budget=owner) == ordinary
    assert guard.estimate_request_tokens(two, turn_budget=owner) == combined
    assert owner.text_used["tokenized"] == paid
    guard.estimate_request_tokens(one, turn_budget=owner)
    assert owner.text_used["tokenized"] > paid


def test_pinned_ordinary_estimate_cannot_be_replaced():
    owner = budget()
    prepay(PAYLOAD, owner, pin=True)
    with pytest.raises((ValueError, guard.ContextLimited)):
        prepay({"content": "Different fallback."}, owner, pin=True)


def test_identical_combined_request_reuses_and_keeps_the_pinned_estimate():
    owner = budget()
    first = prepay(PAYLOAD, owner, pin=True)
    paid = owner.text_used["tokenized"]
    assert prepay(PAYLOAD, owner) == first
    assert owner.text_used["tokenized"] == paid
    prepay({"content": "Later combined request."}, owner)
    before = owner.text_used["tokenized"]
    assert guard.estimate_request_tokens(PAYLOAD, turn_budget=owner) == first
    assert owner.text_used["tokenized"] == before


def test_untrusted_budget_facade_cannot_register_paid_work():
    class Facade:
        def reserve_text(self, *args, **kwargs):
            return True
        def can_publish(self):
            return True

    with pytest.raises((TypeError, ValueError)):
        prepay(PAYLOAD, Facade(), pin=True)


@pytest.mark.parametrize("phase", ["lookup", "prepay"])
def test_clock_callback_cannot_reuse_a_now_changed_tokenizer_policy(monkeypatch, phase):
    armed = [False]
    original = evidence_tokens._ENCODING

    class ChangedPolicy:
        def encode(self, *args, **kwargs):
            return original.encode(*args, **kwargs)

    def clock():
        if armed[0]:
            armed[0] = False
            monkeypatch.setattr(evidence_tokens, "_ENCODING", ChangedPolicy())
        return 0.0

    owner = budget(clock=clock)
    prepay(PAYLOAD, owner, pin=True)
    armed[0] = True
    with pytest.raises(guard.ContextLimited, match="policy_changed"):
        if phase == "lookup":
            guard.estimate_request_tokens(PAYLOAD, turn_budget=owner)
        else:
            prepay(PAYLOAD, owner, pin=True)


@pytest.mark.parametrize("synthesis", [False, True])
def test_clock_callback_cancellation_is_rechecked_before_paid_hit(synthesis):
    armed = [False]

    def clock():
        if armed[0]:
            armed[0] = False
            owner.close("cancelled")
        return 0.0

    owner = budget(clock=clock)
    prepay(PAYLOAD, owner, pin=True)
    active = lease(owner) if synthesis else owner
    armed[0] = True
    with pytest.raises((guard.ContextLimited, asyncio.CancelledError)):
        guard.estimate_request_tokens(PAYLOAD, turn_budget=active)
