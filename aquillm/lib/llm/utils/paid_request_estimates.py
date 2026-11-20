"""Two turn-owned paid tokenizer results; no authorization decisions are cached."""

from dataclasses import dataclass, field

from lib.retrieval.synthesis_budget import SynthesisLease
from lib.retrieval.turn_budget import TurnBudget


@dataclass(frozen=True, slots=True)
class _Estimate:
    encoded: str = field(repr=False)
    policy: tuple = field(repr=False)
    tokens: int
    charged_total: int


@dataclass(frozen=True, slots=True)
class _PaidEstimates:
    owner: object = field(repr=False)
    lock: object = field(repr=False)
    ordinary: _Estimate | None = field(default=None, repr=False)
    combined: _Estimate | None = field(default=None, repr=False)


def _owner(ledger):
    if type(ledger) is TurnBudget:
        return ledger
    if type(ledger) is SynthesisLease and type(ledger.budget) is TurnBudget:
        return ledger.budget if ledger.budget._synthesis is ledger else None
    return None


def _same_policy(left, right):
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


def _matching(cache, encoded, policy):
    return next((entry for entry in (cache.ordinary, cache.combined)
        if entry is not None and entry.encoded == encoded
        and _same_policy(entry.policy, policy)), None)


def _check(cache, ledger):
    owner = cache.owner
    live = ledger.can_publish()
    owner.check_active()
    if (not live or _owner(ledger) is not owner
            or (type(ledger) is TurnBudget and owner.stop_reason is not None)
            or owner._lock is not cache.lock
            or getattr(owner, "_paid_request_estimates", None) is not cache):
        raise ValueError("request_tokenization_limit")


def lookup_paid_estimate(ledger, encoded, policy, policy_now):
    owner = _owner(ledger)
    if owner is None:
        return None
    with owner._lock:
        cache = getattr(owner, "_paid_request_estimates", None)
        if type(cache) is not _PaidEstimates or cache.owner is not owner:
            return None
        entry = _matching(cache, encoded, policy)
        if entry is None:
            return None
        _check(cache, ledger)
        if not _same_policy(policy, policy_now()):
            raise ValueError("request_estimate_policy_changed")
        if owner._text["tokenized"] < entry.charged_total:
            raise ValueError("request_tokenization_limit")
        return entry.tokens


def prepay_estimate(encoded, *, ledger, pin, policy, policy_now, tokenize):
    if type(ledger) is not TurnBudget or type(pin) is not bool:
        raise TypeError("paid request estimates require the original turn budget")
    with ledger._lock:
        cache = getattr(ledger, "_paid_request_estimates", None)
        if cache is None:
            cache = _PaidEstimates(ledger, ledger._lock)
            ledger._paid_request_estimates = cache
        if type(cache) is not _PaidEstimates or cache.owner is not ledger:
            raise ValueError("request_estimate_owner_changed")
        _check(cache, ledger)
        if not _same_policy(policy, policy_now()):
            raise ValueError("request_estimate_policy_changed")
        entry = _matching(cache, encoded, policy)
        if pin and cache.ordinary is not None and entry is not cache.ordinary:
            raise ValueError("request_estimate_pin_changed")
        if entry is None:
            if not ledger.reserve_text(len(encoded), kind="tokenized"):
                raise ValueError("request_tokenization_limit")
            _check(cache, ledger)
            tokens = tokenize(encoded)
            _check(cache, ledger)
            if not _same_policy(policy, policy_now()):
                raise ValueError("request_estimate_policy_changed")
            entry = _Estimate(encoded, policy, tokens, ledger._text["tokenized"])
        elif ledger._text["tokenized"] < entry.charged_total:
            raise ValueError("request_tokenization_limit")
        if pin:
            object.__setattr__(cache, "ordinary", entry)
            if cache.combined is entry:
                object.__setattr__(cache, "combined", None)
        elif entry is not cache.ordinary:
            object.__setattr__(cache, "combined", entry)
        return entry.tokens
