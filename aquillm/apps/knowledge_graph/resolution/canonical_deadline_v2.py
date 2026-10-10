"""PostgreSQL statement/lock bounds decrease against one absolute deadline."""

from contextlib import contextmanager
from math import isfinite

from django.db import connections, transaction


class ReconciliationDeadlineExceeded(TimeoutError):
    pass


def check_deadline(deadline, clock):
    if type(deadline) not in (int, float) or not isfinite(deadline):
        raise ValueError("deadline must be finite monotonic time")
    remaining = int((deadline - clock()) * 1000)
    if remaining < 1:
        raise ReconciliationDeadlineExceeded()
    return remaining


@contextmanager
def bounded_transaction(*, using, deadline, clock):
    connection = connections[using]
    if connection.vendor != "postgresql":
        raise RuntimeError("bounded reconciliation requires PostgreSQL")
    check_deadline(deadline, clock)
    with transaction.atomic(using=using):
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT name, setting::bigint FROM pg_settings "
                "WHERE name IN ('lock_timeout', 'statement_timeout')"
            )
            operator = dict(cursor.fetchall())

        def bounded_execute(execute, sql, params, many, context):
            remaining = check_deadline(deadline, clock)
            limits = [
                min(remaining, operator[name]) if operator[name] else remaining
                for name in ("lock_timeout", "statement_timeout")
            ]
            # A query may use a named server cursor, which can execute only once.
            # A separate unnamed control cursor also avoids wrapper recursion.
            with connection.connection.cursor() as control:
                control.execute(
                    "SELECT set_config('lock_timeout', %s, true), "
                    "set_config('statement_timeout', %s, true)",
                    [f"{limits[0]}ms", f"{limits[1]}ms"],
                )
            result = execute(sql, params, many, context)
            check_deadline(deadline, clock)
            return result

        with connection.execute_wrapper(bounded_execute):
            yield
            check_deadline(deadline, clock)
