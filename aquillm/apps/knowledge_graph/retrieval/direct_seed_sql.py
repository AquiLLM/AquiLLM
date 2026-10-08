"""Transaction-scoped budgets for fully consumed direct seed reads."""

from contextlib import contextmanager
from math import floor
from time import monotonic

from django.db import DatabaseError, connections, transaction


class DirectSeedReadTimeout(TimeoutError):
    """The direct branch read budget expired or PostgreSQL canceled its read."""


def check_seed_deadline(deadline, *, clock=monotonic):
    if deadline is not None and clock() >= deadline:
        raise DirectSeedReadTimeout("direct seed read deadline expired")


@contextmanager
def bounded_seed_read(*, using, deadline=None, alias=False, clock=monotonic):
    check_seed_deadline(deadline, clock=clock)
    connection = connections[using]
    if connection.vendor != "postgresql":
        yield
        check_seed_deadline(deadline, clock=clock)
        return
    try:
        # Rollback to this savepoint also restores SET LOCAL on SQL/Python errors.
        with transaction.atomic(using=using):
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT current_setting('statement_timeout'), "
                    "current_setting('join_collapse_limit'), "
                    "(SELECT setting::bigint FROM pg_settings "
                    "WHERE name = 'statement_timeout')"
                )
                timeout, joins, prior_ms = cursor.fetchone()
                if deadline is not None:
                    remaining_ms = floor((deadline - clock()) * 1000)
                    if remaining_ms < 1:
                        raise DirectSeedReadTimeout("direct seed read deadline expired")
                    effective_ms = (
                        min(remaining_ms, prior_ms) if prior_ms else remaining_ms
                    )
                    cursor.execute(
                        "SELECT set_config('statement_timeout', %s, true)",
                        [str(effective_ms)],
                    )
                if alias:
                    cursor.execute("SET LOCAL join_collapse_limit=1")
            check_seed_deadline(deadline, clock=clock)
            yield
            check_seed_deadline(deadline, clock=clock)
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true), "
                    "set_config('join_collapse_limit', %s, true)",
                    [timeout, joins],
                )
    except DatabaseError as error:
        cause = error.__cause__
        if (
            getattr(cause, "sqlstate", None) == "57014"
            or getattr(cause, "pgcode", None) == "57014"
        ):
            raise DirectSeedReadTimeout("direct seed database read canceled") from error
        raise
