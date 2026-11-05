"""Invalidate late thread publications without discarding earlier evidence."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import isfinite
from time import monotonic

_OPERATION = ContextVar("retrieval_operation", default=None)


@dataclass
class Operation:
    budget: object
    live: bool = True
    deadline: float | None = None
    clock: object = monotonic
    parent: object = None
    source_limit: int | None = None

    def __setattr__(self, name, value):
        # Captured native owners retain this exact operation. Its admission
        # ceiling must not widen while workers are using that captured owner.
        if name == "source_limit" and name in self.__dict__:
            raise AttributeError("operation source limit is immutable")
        object.__setattr__(self, name, value)

    def active(self):
        return (
            self.live
            and (self.deadline is None or self.clock() < self.deadline)
            and (self.parent is None or self.parent.active())
            and self.live  # Clock/parent callbacks may revoke during this check.
        )

    def remaining_ms(self):
        if not self.active():
            return 0.0
        remaining = self.budget.remaining_ms()
        if self.deadline is not None:
            remaining = min(remaining, max(0.0, (self.deadline - self.clock()) * 1000))
        if self.parent is not None:
            remaining = min(remaining, self.parent.remaining_ms())
        return remaining if self.active() else 0.0

    def deadline_for(self, clock, *, reserve_ms=0):
        """Translate remaining original allowance; never compare clock epochs."""
        if type(reserve_ms) is not int or reserve_ms < 0:
            raise ValueError("reserve must be a nonnegative exact integer")
        now = clock()  # Sampling before remaining can only narrow this bound.
        return now + max(0.0, self.remaining_ms() - reserve_ms) / 1000


def operation_live(budget):
    operation = _OPERATION.get()
    return operation is None or operation.budget is not budget or operation.active()


def operation_source_limit(budget):
    """Action-local cumulative ceiling; a child may only narrow its parent."""
    operation = _OPERATION.get()
    ceiling = budget.limits.unique_sources
    while operation is not None and operation.budget is budget:
        if operation.source_limit is not None:
            ceiling = min(ceiling, operation.source_limit)
        operation = operation.parent
    return ceiling


def capture_operation_bound(budget):
    operation = _OPERATION.get()
    if operation is None:
        return None
    if type(operation) is not Operation or operation.budget is not budget:
        raise ValueError("operation budget identity changed")
    if not operation.active() or not budget.can_publish():
        raise TimeoutError("operation is inactive")
    return operation if operation.deadline is not None else None


@contextmanager
def operation_scope(budget, *, timeout_ms=None, source_limit=None):
    if source_limit is not None and (type(source_limit) is not int or source_limit < 0):
        raise ValueError("source limit must be a nonnegative exact integer")
    parent = _OPERATION.get()
    if parent is not None and parent.budget is not budget:
        if parent.deadline is not None or parent.source_limit is not None:
            raise ValueError("bounded operation budget identity changed")
        parent = None
    clock = getattr(budget, "_clock", monotonic)
    deadline = None
    if timeout_ms is not None:
        if type(timeout_ms) not in (int, float) or not isfinite(timeout_ms) or timeout_ms <= 0:
            raise ValueError("operation timeout must be positive and finite")
        deadline = clock() + min(timeout_ms, budget.remaining_ms()) / 1000
    if parent is not None and parent.deadline is not None:
        parent_deadline = parent.deadline_for(clock)
        deadline = parent_deadline if deadline is None else min(deadline, parent_deadline)
    if parent is not None and parent.source_limit is not None:
        source_limit = parent.source_limit if source_limit is None else min(
            source_limit, parent.source_limit
        )
    operation = Operation(
        budget, deadline=deadline, clock=clock, parent=parent, source_limit=source_limit
    )
    token = _OPERATION.set(operation)
    try:
        yield operation
    finally:
        with budget._lock:
            operation.live = False
        _OPERATION.reset(token)
