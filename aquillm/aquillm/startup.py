"""Process-local controls for narrowly scoped Django startup modes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_OFFLINE_EVALUATION_STARTUP: ContextVar[bool] = ContextVar(
    "aquillm_offline_evaluation_startup", default=False
)


def is_offline_evaluation_startup() -> bool:
    """Return whether Django is currently bootstrapping for the offline evaluator."""

    return _OFFLINE_EVALUATION_STARTUP.get()


@contextmanager
def offline_evaluation_startup() -> Iterator[None]:
    """Disable external startup side effects only around offline Django setup."""

    token = _OFFLINE_EVALUATION_STARTUP.set(True)
    try:
        yield
    finally:
        _OFFLINE_EVALUATION_STARTUP.reset(token)
