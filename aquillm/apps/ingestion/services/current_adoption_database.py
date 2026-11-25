"""Database statements share the current-adoption operation deadline."""
from contextlib import contextmanager
import time
from django.db import DatabaseError
from apps.documents.services.representation_contracts import SourceScopeUnavailable


def remaining_ms(deadline):
    milliseconds = int((deadline - time.monotonic()) * 1000)
    if milliseconds <= 0:
        raise SourceScopeUnavailable("source_scope_unavailable")
    return milliseconds


@contextmanager
def bounded_database(connection, deadline):
    """Bound each SQL statement without holding a transaction across source I/O."""
    remaining_ms(deadline)
    with connection.cursor() as cursor:
        cursor.execute("SELECT name, setting::bigint FROM pg_settings "
                       "WHERE name IN ('statement_timeout', 'lock_timeout')")
        previous = dict(cursor.fetchall())

    def bounded(execute, sql, params, many, context):
        # Savepoint cleanup must remain possible after cancellation.
        if sql.startswith(("ROLLBACK TO SAVEPOINT", "RELEASE SAVEPOINT")):
            return execute(sql, params, many, context)
        remaining = remaining_ms(deadline)
        statement = min(previous["statement_timeout"] or remaining, remaining)
        lock = min(previous["lock_timeout"] or remaining, remaining, 5000)
        execute("SELECT set_config('statement_timeout', %s, %s), set_config('lock_timeout', %s, %s)",
                [str(statement), connection.in_atomic_block, str(lock), connection.in_atomic_block],
                False, context)
        remaining_ms(deadline)
        result = execute(sql, params, many, context)
        remaining_ms(deadline)
        return result

    try:
        with connection.execute_wrapper(bounded):
            yield
    finally:
        if not connection.needs_rollback:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT set_config('statement_timeout', %s, false), "
                                   "set_config('lock_timeout', %s, false)",
                                   [str(previous["statement_timeout"]), str(previous["lock_timeout"])])
            except DatabaseError:
                connection.close()
