"""Pure, bounded v1 policy for selecting a PageRank restart probability.

Colon-bearing and multiline questions deliberately remain balanced.  The current
query builder can prefix a retrieved title without preserving a context marker,
so guessing at title boundaries would be unsafe.  This conservatively abstains
on legitimate identifiers containing colons; callers can measure that coverage
cost before introducing an explicit query-context contract.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Literal, final

from .topology.contracts import HybridBranchKind

_INTENTS = frozenset(("focused", "balanced", "relational"))
_SUPPORT_STATUSES = frozenset(("supported", "insufficient", "unknown"))
_REASONS = frozenset(
    (
        "fixed_mode",
        "focused_supported",
        "relational_supported",
        "balanced_intent",
        "unknown_signals",
        "seed_support_insufficient",
        "cap_pressure",
        "insufficient_connections",
    )
)
_RESTARTS = frozenset((0.15, 0.20, 0.35))
_HEX_DIGEST = re.compile(r"[0-9a-fA-F]{64}\Z")


@final
@dataclass(frozen=True, slots=True)
class PPRPolicySignalsV1:
    branch_kind: HybridBranchKind
    intent: Literal["focused", "balanced", "relational"]
    seed_count: int
    support_status: Literal["supported", "insufficient", "unknown"]
    cap_pressure: bool
    outward_seed_mass: float
    support_digest: str

    def __post_init__(self) -> None:
        if type(self.branch_kind) is not HybridBranchKind:
            raise TypeError("branch_kind must be an exact HybridBranchKind")
        if type(self.intent) is not str:
            raise TypeError("intent must be an exact string")
        if self.intent not in _INTENTS:
            raise ValueError("intent is unsupported")
        if type(self.seed_count) is not int:
            raise TypeError("seed_count must be an exact integer")
        if not 1 <= self.seed_count <= 64:
            raise ValueError("seed_count must be in [1, 64]")
        if type(self.support_status) is not str:
            raise TypeError("support_status must be an exact string")
        if self.support_status not in _SUPPORT_STATUSES:
            raise ValueError("support_status is unsupported")
        if type(self.cap_pressure) is not bool:
            raise TypeError("cap_pressure must be an exact boolean")
        if type(self.outward_seed_mass) is not float:
            raise TypeError("outward_seed_mass must be an exact float")
        if not (
            isfinite(self.outward_seed_mass)
            and 0.0 <= self.outward_seed_mass <= 1.0
        ):
            raise ValueError("outward_seed_mass must be finite and in [0, 1]")
        if type(self.support_digest) is not str:
            raise TypeError("support_digest must be an exact string")
        if _HEX_DIGEST.fullmatch(self.support_digest) is None:
            raise ValueError("support_digest must be a 64-hex digest")


@final
@dataclass(frozen=True, slots=True)
class PPRPolicyDecisionV1:
    policy_version: str
    restart: float
    iterations: int
    reason: str

    def __post_init__(self) -> None:
        if type(self.policy_version) is not str:
            raise TypeError("policy_version must be an exact string")
        if self.policy_version != "ppr_restart_policy_v1":
            raise ValueError("policy_version is unsupported")
        if type(self.restart) is not float:
            raise TypeError("restart must be an exact float")
        if self.restart not in _RESTARTS:
            raise ValueError("restart is unsupported")
        if type(self.iterations) is not int:
            raise TypeError("iterations must be an exact integer")
        if self.iterations != 8:
            raise ValueError("iterations must be eight")
        if type(self.reason) is not str:
            raise TypeError("reason must be an exact string")
        if self.reason not in _REASONS:
            raise ValueError("reason is unsupported")


def classify_ppr_intent(query: str, *, context_ambiguous: bool = False) -> str:
    if type(query) is not str or type(context_ambiguous) is not bool:
        raise TypeError("query and context_ambiguous must have exact types")
    if context_ambiguous or ":" in query or "\n" in query:
        return "balanced"
    text = " ".join(query.casefold().split())
    relational = (
        r"\b(?:relationship between|connection between|connected to|"
        r"path between|linked through)\b"
    )
    if re.search(relational, text):
        return "relational"
    if re.match(r"^(?:who|when|where|how many|define)\b", text):
        return "focused"
    return "balanced"


def choose_ppr_restart(
    signals: PPRPolicySignalsV1, *, mode: str
) -> PPRPolicyDecisionV1:
    if type(signals) is not PPRPolicySignalsV1:
        raise TypeError("signals must be an exact PPRPolicySignalsV1")
    if mode not in ("fixed", "shadow", "adaptive"):
        raise ValueError("unsupported restart mode")
    reason, restart = "balanced_intent", 0.20
    if mode == "fixed":
        reason = "fixed_mode"
    elif signals.support_status == "unknown":
        reason = "unknown_signals"
    elif signals.support_status == "insufficient":
        reason = "seed_support_insufficient"
    elif signals.cap_pressure:
        reason = "cap_pressure"
    elif signals.intent == "focused":
        reason, restart = "focused_supported", 0.35
    elif signals.intent == "relational":
        if signals.seed_count >= 2 and signals.outward_seed_mass >= 0.50:
            reason, restart = "relational_supported", 0.15
        else:
            reason = "insufficient_connections"
    return PPRPolicyDecisionV1("ppr_restart_policy_v1", restart, 8, reason)


def ppr_policy_input_digest(signals: PPRPolicySignalsV1) -> str:
    """Return a canonical digest of the bounded inputs to the policy."""
    if type(signals) is not PPRPolicySignalsV1:
        raise TypeError("signals must be an exact PPRPolicySignalsV1")
    payload = {
        "branch_kind": signals.branch_kind.value,
        "intent": signals.intent,
        "seed_count": signals.seed_count,
        "support_status": signals.support_status,
        "cap_pressure": signals.cap_pressure,
        "outward_seed_mass": signals.outward_seed_mass.hex(),
        "support_digest": signals.support_digest,
    }
    encoded = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
