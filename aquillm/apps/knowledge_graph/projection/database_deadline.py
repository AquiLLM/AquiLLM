"""One original projection deadline, including restricted PostgreSQL execution."""

from contextlib import contextmanager
from math import isfinite
from threading import Lock
from time import monotonic

from django.db import connections, transaction

from apps.knowledge_graph.database_cursor_deadline import bounded_cursor_calls

from .operation_context import _CURRENT, current_operation


class ProjectionOperation:
    def __init__(self, timeout_ms, *, deadline=None, call_timeout_ms=None):
        if type(timeout_ms) is not int or not 0 < timeout_ms <= 300000:
            raise ValueError("projection timeout must be positive milliseconds")
        self._lock = Lock()
        self._closed = False
        self._call_timeout_ms = None
        self._deadline = monotonic() + timeout_ms / 1000
        self.limit_calls(call_timeout_ms)
        if deadline is not None:
            self.tighten(deadline)

    def tighten(self, deadline):
        if type(deadline) not in (int, float) or not isfinite(deadline):
            raise ValueError("projection deadline must be finite")
        with self._lock:
            self._deadline = min(self._deadline, deadline)

    def remaining_ms(self):
        with self._lock:
            if self._closed:
                raise TimeoutError("projection_operation_revoked")
            remaining = int((self._deadline - monotonic()) * 1000)
        if remaining <= 0:
            raise TimeoutError("projection_database_deadline")
        return remaining

    def close(self):
        with self._lock:
            self._closed = True

    def limit_calls(self, timeout_ms):
        if timeout_ms is None:
            return
        if type(timeout_ms) is not int or not 0 < timeout_ms <= 5000:
            raise ValueError("projection call timeout must be within 5000 milliseconds")
        with self._lock:
            self._call_timeout_ms = min(self._call_timeout_ms or timeout_ms, timeout_ms)

    def call_remaining_ms(self):
        remaining = self.remaining_ms()
        with self._lock:
            return min(remaining, self._call_timeout_ms or remaining)

    def call(self, ceiling_ms=None):
        return _ProjectionCall(self, ceiling_ms)

    def end(self):
        with self._lock:
            return self._deadline


class _ProjectionCall:
    """One nonrenewing call, retaining its original job even across aliases."""

    def __init__(self, operation, ceiling_ms):
        self.operation = operation
        self.started = monotonic()
        self.deadline = self.started + operation.call_remaining_ms() / 1000
        if ceiling_ms:
            self.limit(ceiling_ms)

    def limit(self, ceiling_ms):
        self.deadline = min(self.deadline, self.started + ceiling_ms / 1000)

    def remaining_ms(self):
        remaining = min(
            self.operation.remaining_ms(), int((self.deadline - monotonic()) * 1000)
        )
        if remaining <= 0:
            raise TimeoutError("projection_database_call_deadline")
        return remaining


@contextmanager
def projection_operation(
    timeout_ms=None, *, operation=None, deadline=None, call_timeout_ms=None
):
    if timeout_ms is not None and (type(timeout_ms) is not int or timeout_ms <= 0):
        raise ValueError("projection timeout must be positive milliseconds")
    if operation is not None and type(operation) is not ProjectionOperation:
        raise TypeError("projection operation must be exact")
    outer = current_operation()
    if outer is not None and operation is not None and outer is not operation:
        raise ValueError("projection operation was replaced")
    active = outer or operation
    owned = active is None
    if active is None:
        if timeout_ms is None:
            from .runtime import load_projection_runtime_settings

            timeout_ms = load_projection_runtime_settings().projection_timeout_ms
        active = ProjectionOperation(
            timeout_ms, deadline=deadline, call_timeout_ms=call_timeout_ms
        )
    elif deadline is not None:
        active.tighten(deadline)
    if timeout_ms is not None:
        active.tighten(monotonic() + timeout_ms / 1000)
    active.limit_calls(call_timeout_ms)
    active.remaining_ms()
    token = _CURRENT.set(active)
    try:
        yield active
        active.remaining_ms()
    finally:
        if owned:
            active.close()
        _CURRENT.reset(token)


@contextmanager
def projection_database(using, *, operation=None):
    with projection_operation(operation=operation) as active:
        setup = active.call()
        connection = connections[using]
        with transaction.atomic(using=using):
            # Read inherited settings before enclosing execute/fetch guards can
            # reinstall their own limits on this nested scope's metadata read.
            with (
                connection.wrap_database_errors,
                connection.connection.cursor() as cursor,
            ):
                cursor.execute(
                    "SELECT name, setting::bigint FROM pg_settings "
                    "WHERE name IN ('lock_timeout','statement_timeout')"
                )
                previous = dict(cursor.fetchall())
            if previous["statement_timeout"]:
                setup.limit(previous["statement_timeout"])
            setup.remaining_ms()
            complete = False
            try:
                with bounded_cursor_calls(
                    connection, active, previous
                ) as bounded_execute:
                    with connection.execute_wrapper(bounded_execute):
                        yield active
                        active.remaining_ms()
                        complete = True
            finally:
                # On error the atomic/savepoint rollback restores SET LOCAL values.
                if complete and not connection.needs_rollback:
                    try:
                        cleanup = active.call(previous["statement_timeout"])
                        with connection.connection.cursor() as control:
                            control.execute(
                                "SELECT set_config('lock_timeout',%s,true),"
                                "set_config('statement_timeout',%s,true)",
                                [
                                    f"{previous['lock_timeout']}ms",
                                    f"{previous['statement_timeout']}ms",
                                ],
                            )
                        cleanup.remaining_ms()
                    except Exception:
                        connection.close()
                        raise


def bounded_state_call(method):
    """Keep iteration/fetch and nested alias calls on one original operation."""
    from functools import wraps

    @wraps(method)
    def run(self, *args, **kwargs):
        with projection_operation(operation=self.operation):
            return method(self, *args, **kwargs)

    return run
