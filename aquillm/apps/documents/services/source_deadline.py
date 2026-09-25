"""Bound source preparation and finalization without renewing the turn ledger."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import monotonic

_DEADLINE = ContextVar("source_preparation_deadline", default=None)


def source_remaining_ms(default):
    current = _DEADLINE.get()
    if current is None:
        return default
    deadline, clock, budget = current
    remaining = max(0, int((deadline - clock()) * 1000))
    if budget is not None:
        remaining = min(remaining, budget.remaining_ms())
    return min(default, remaining)


def check_source_deadline():
    from .source_loading import SourcePreparationLimited

    if source_remaining_ms(1) <= 0:
        raise SourcePreparationLimited("source preparation deadline")


@contextmanager
def source_preparation_scope(deadline, clock, budget=None):
    parent = _DEADLINE.get()
    if parent is not None:
        deadline = min(deadline, parent[0])
        budget = parent[2]
    token = _DEADLINE.set((deadline, clock, budget))
    try:
        check_source_deadline()
        yield
        check_source_deadline()
    finally:
        _DEADLINE.reset(token)


def bound_source_finalization(function):
    """Post-score authority checks remain within the original retrieval allowance.

    The caller separately bounds initial hydration by its scoring/preparation
    deadline. Expiring that phase cannot extend scoring or discard a frozen pool
    while the original ledger still permits its current-source revalidation.
    """

    @wraps(function)
    def bounded(*args, **kwargs):
        if not kwargs.get("source_mode"):
            return function(*args, **kwargs)
        clock = kwargs.get("clock", monotonic)
        budget = kwargs.get("turn_budget")
        deadline = (
            clock() + budget.remaining_ms() / 1000
            if budget is not None
            else kwargs["deadline"]
        )
        with source_preparation_scope(deadline, clock, budget):
            return function(*args, **kwargs)

    return bounded
