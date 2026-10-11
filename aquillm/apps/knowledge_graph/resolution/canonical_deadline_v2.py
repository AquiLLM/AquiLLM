"""PostgreSQL statement/lock bounds decrease against one absolute deadline."""

from contextlib import contextmanager
from math import isfinite

from django.db import connections, transaction

from apps.knowledge_graph.database_cursor_deadline import bounded_cursor_calls


class ReconciliationDeadlineExceeded(TimeoutError):
    pass


def check_deadline(deadline, clock):
    if type(deadline) not in (int, float) or not isfinite(deadline):
        raise ValueError("deadline must be finite monotonic time")
    remaining = int((deadline - clock()) * 1000)
    if remaining < 1:
        raise ReconciliationDeadlineExceeded()
    return remaining


class _OriginalDeadline:
    def __init__(self, deadline, clock):
        self.deadline, self.clock = deadline, clock

    def remaining_ms(self):
        return check_deadline(self.deadline, self.clock)


@contextmanager
def bounded_transaction(*, using, deadline, clock):
    connection = connections[using]
    if connection.vendor != "postgresql":
        raise RuntimeError("bounded reconciliation requires PostgreSQL")
    check_deadline(deadline, clock)
    with transaction.atomic(using=using):
        with connection.wrap_database_errors, connection.connection.cursor() as cursor:
            cursor.execute(
                "SELECT name, setting::bigint FROM pg_settings "
                "WHERE name IN ('lock_timeout', 'statement_timeout')"
            )
            operator = dict(cursor.fetchall())

        complete = False
        try:
            active = _OriginalDeadline(deadline, clock)
            with bounded_cursor_calls(connection, active, operator) as bounded_execute:
                with connection.execute_wrapper(bounded_execute):
                    yield
                    check_deadline(deadline, clock)
                    complete = True
        finally:
            # Atomic rollback restores SET LOCAL after failure. Success inside a
            # caller's transaction must restore it explicitly before release.
            if complete and not connection.needs_rollback:
                try:
                    with connection.wrap_database_errors, connection.connection.cursor() as control:
                        control.execute("SELECT set_config('lock_timeout',%s,true),"
                                        "set_config('statement_timeout',%s,true)",
                                        [f"{operator['lock_timeout']}ms",
                                         f"{operator['statement_timeout']}ms"])
                except Exception:
                    connection.close()
                    raise
