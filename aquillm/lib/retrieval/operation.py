"""Invalidate late thread publications without discarding earlier evidence."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_OPERATION = ContextVar("retrieval_operation", default=None)


@dataclass
class Operation:
    budget: object
    live: bool = True


def operation_live(budget):
    operation = _OPERATION.get()
    return operation is None or operation.budget is not budget or operation.live


@contextmanager
def operation_scope(budget):
    operation = Operation(budget)
    token = _OPERATION.set(operation)
    try:
        yield operation
    finally:
        with budget._lock:
            operation.live = False
        _OPERATION.reset(token)
