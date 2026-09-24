"""Opt-in exact ledger transitions for isolated operational evaluation."""

from dataclasses import asdict
from functools import wraps

from lib.evidence_observation import active, publish


def snapshot(budget):
    return {
        "actions": len(budget._actions),
        "sources": len(budget._sources),
        "pairs": budget._pairs.copy(),
        "text": budget._text.copy(),
        "planner_calls": budget._planner_calls,
        "inflight": budget._inflight,
        "closed": budget._closed_reason,
        "terminal": budget._terminal_reason,
        "remaining_ms": max(0, int((budget._deadline - budget._clock()) * 1000)),
    }


def observed_budget(cls):
    """No sink means no snapshots, clock reads, locks or callbacks are added."""

    def wrap(name, method):
        @wraps(method)
        def call(self, *args, **kwargs):
            if not active():
                return method(self, *args, **kwargs)
            if name == "__init__":
                result = method(self, *args, **kwargs)
                publish(
                    "ledger_start",
                    {
                        "ledger_id": str(id(self)),
                        "limits": asdict(self.limits),
                        "state": snapshot(self),
                    },
                )
                return result
            with self._lock:
                before = snapshot(self)
                result = method(self, *args, **kwargs)
                publish(
                    "ledger",
                    {
                        "ledger_id": str(id(self)),
                        "operation": name,
                        "arguments": [str(a) for a in args],
                        "keywords": kwargs,
                        "before": before,
                        "after": snapshot(self),
                        "result": result,
                    },
                )
                return result

        return call

    for name in (
        "__init__",
        "reserve_action",
        "reserve_planner",
        "admit_source",
        "reserve_pairs",
        "reserve_text",
        "can_start_optional",
        "close",
        "start_pair",
        "finish_pair",
        "can_publish",
        "scoring_remaining_ms",
    ):
        setattr(cls, name, wrap(name, getattr(cls, name)))
    return cls
