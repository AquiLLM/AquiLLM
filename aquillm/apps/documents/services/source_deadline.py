"""Propagate the final preparation deadline through nested source operations."""

from contextvars import ContextVar
from functools import wraps
from time import monotonic

_DEADLINE = ContextVar("source_preparation_deadline", default=None)


def source_remaining_ms(default):
    current = _DEADLINE.get()
    if current is None:
        return default
    deadline, clock = current
    return min(default, max(0, int((deadline - clock()) * 1000)))


def check_source_deadline():
    from .source_loading import SourcePreparationLimited

    if source_remaining_ms(1) <= 0:
        raise SourcePreparationLimited("source preparation deadline")


def bound_source_preparation(function):
    @wraps(function)
    def bounded(*args, **kwargs):
        if not kwargs.get("source_mode"):
            return function(*args, **kwargs)
        token = _DEADLINE.set((kwargs["deadline"], kwargs.get("clock", monotonic)))
        try:
            check_source_deadline()
            result = function(*args, **kwargs)
            check_source_deadline()
            return result
        finally:
            _DEADLINE.reset(token)

    return bounded
