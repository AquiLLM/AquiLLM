"""Django-independent carrier for the original projection operation."""
from contextvars import ContextVar

_CURRENT = ContextVar("projection_database_operation", default=None)


def current_operation():
    return _CURRENT.get()
