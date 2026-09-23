"""Atomic cumulative pilot limits for one retrieval turn.

All fallbacks must receive the same TurnBudget instance. A closed or expired
ledger fences late retrieval and cache publication; already frozen validated
evidence can still be used for answer synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Callable, Hashable

from .evidence import SourceIdentity

_PILOT_MAX = {
    "actions": 3, "unique_sources": 45,
    "materialized_codepoints": 250_000, "tokenized_codepoints": 1_000_000,
    "acquisition_pairs": 90, "final_pairs": 45, "in_flight_pairs": 6,
    "planner_calls": 2, "planner_call_ms": 2_000,
    "planner_output_tokens": 512, "retrieval_ms": 15_000,
    "final_scoring_ms": 3_000,
}


@dataclass(frozen=True, slots=True)
class TurnLimits:
    actions: int = 3
    unique_sources: int = 45
    materialized_codepoints: int = 250_000
    tokenized_codepoints: int = 1_000_000
    acquisition_pairs: int = 90
    final_pairs: int = 45
    in_flight_pairs: int = 6
    planner_calls: int = 2
    planner_call_ms: int = 2_000
    planner_output_tokens: int = 512
    retrieval_ms: int = 15_000
    final_scoring_ms: int = 3_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _PILOT_MAX[name]:
                raise ValueError(f"{name} must be within the pilot allowance")
        if self.acquisition_pairs + self.final_pairs > 135:
            raise ValueError("overall pair allowance exceeds 135")
        if self.final_scoring_ms > self.retrieval_ms:
            raise ValueError("final scoring reserve exceeds retrieval deadline")

    def clamped_to(self, parent: TurnLimits) -> TurnLimits:
        """A nested caller can narrow, never enlarge, a parent allowance."""
        return TurnLimits(**{
            name: min(getattr(self, name), getattr(parent, name))
            for name in self.__dataclass_fields__
        })


class TurnBudget:
    def __init__(self, limits: TurnLimits, *, clock: Callable[[], float] = monotonic):
        self.limits = limits
        self._clock = clock
        self._deadline = clock() + limits.retrieval_ms / 1000
        self._lock = RLock()
        self._closed_reason: str | None = None
        self._actions: set[str] = set()
        self._sources: set[Hashable] = set()
        self._pairs = {"acquisition": 0, "final": 0}
        self._text = {"materialized": 0, "tokenized": 0}

    def _open(self) -> bool:
        if self._closed_reason is not None:
            return False
        if self._clock() >= self._deadline:
            self._closed_reason = "deadline"
            return False
        return True

    @property
    def stop_reason(self) -> str | None:
        with self._lock:
            return self._closed_reason

    @property
    def pairs_used(self) -> dict[str, int]:
        with self._lock:
            return self._pairs.copy()

    @property
    def text_used(self) -> dict[str, int]:
        with self._lock:
            return self._text.copy()

    @property
    def actions_used(self) -> int:
        with self._lock:
            return len(self._actions)

    @property
    def sources_used(self) -> int:
        with self._lock:
            return len(self._sources)

    def reserve_action(self, signature: str) -> bool:
        if not isinstance(signature, str) or not signature:
            raise ValueError("action signature must be nonempty")
        with self._lock:
            if not self._open() or signature in self._actions or len(self._actions) >= self.limits.actions:
                return False
            self._actions.add(signature)
            return True

    def admit_source(self, identity: SourceIdentity | tuple[int, str]) -> bool:
        if isinstance(identity, tuple):
            if len(identity) != 2:
                raise ValueError("source identity requires chunk and revision")
            identity = SourceIdentity(*identity)
        if not isinstance(identity, SourceIdentity):
            raise ValueError("source identity requires chunk and revision")
        with self._lock:
            if not self._open():
                return False
            if identity in self._sources:
                return True
            if len(self._sources) >= self.limits.unique_sources:
                return False
            self._sources.add(identity)
            return True

    def reserve_pairs(self, count: int, *, phase: str) -> bool:
        if phase not in self._pairs or isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("invalid pair reservation")
        with self._lock:
            if not self._open() or self._pairs[phase] + count > getattr(self.limits, f"{phase}_pairs"):
                return False
            self._pairs[phase] += count
            return True

    def reserve_text(self, count: int, *, kind: str) -> bool:
        if kind not in self._text or isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("invalid text reservation")
        with self._lock:
            if not self._open() or self._text[kind] + count > getattr(self.limits, f"{kind}_codepoints"):
                return False
            self._text[kind] += count
            return True

    def remaining_ms(self) -> int:
        with self._lock:
            if not self._open():
                return 0
            return max(0, int((self._deadline - self._clock()) * 1000))

    def can_start_optional(self, worst_case_ms: int, completion_reserve_ms: int) -> bool:
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (worst_case_ms, completion_reserve_ms)):
            raise ValueError("optional work and completion reserve must be nonnegative milliseconds")
        with self._lock:
            return self._open() and (
                worst_case_ms + completion_reserve_ms + self.limits.final_scoring_ms
                <= max(0, int((self._deadline - self._clock()) * 1000))
            )

    def close(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("closure reason must be nonempty")
        with self._lock:
            if self._closed_reason is None:
                self._closed_reason = reason

    def can_publish(self) -> bool:
        with self._lock:
            return self._open()
