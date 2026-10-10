"""Original local build deadlines, including SQL outside publication locks."""

from contextlib import contextmanager
from contextvars import ContextVar
from django.db import connections, transaction, DatabaseError

_ACTIVE = ContextVar("embedding_build_database", default=None)


def check_embedding_deadline(budget):
    milliseconds = int(budget.remaining_seconds() * 1000)
    if milliseconds <= 0:
        raise TimeoutError("embedding_deadline")
    return milliseconds


def check_active_build_deadline():
    active = _ACTIVE.get()
    if active is not None:
        check_embedding_deadline(active[1])


@contextmanager
def embedding_build_database(budget, *, using):
    """Bound every SQL statement without holding a transaction around media I/O."""
    check_embedding_deadline(budget)
    connection = connections[using]
    if _ACTIVE.get() == (connection, budget):
        yield
        check_embedding_deadline(budget)
        return
    previous = 0
    if connection.vendor == "postgresql":
        with connection.cursor() as cursor:
            check_embedding_deadline(budget)
            cursor.execute(
                "SELECT setting FROM pg_settings WHERE name = 'statement_timeout'"
            )
            previous = int(cursor.fetchone()[0])  # pg_settings uses milliseconds.

    def bounded(execute, sql, params, many, context):
        # Transaction cleanup must remain possible after expiry/cancellation.
        if sql.startswith(("ROLLBACK TO SAVEPOINT", "RELEASE SAVEPOINT")):
            return execute(sql, params, many, context)
        remaining = check_embedding_deadline(budget)
        if connection.vendor == "postgresql":
            milliseconds = min(previous, remaining) if previous else remaining
            execute(
                "SELECT set_config('statement_timeout', %s, %s)",
                [f"{milliseconds}ms", connection.in_atomic_block],
                False,
                context,
            )
            check_embedding_deadline(budget)
        result = execute(sql, params, many, context)
        check_embedding_deadline(budget)
        return result

    token = _ACTIVE.set((connection, budget))
    try:
        with connection.execute_wrapper(bounded):
            yield
            check_embedding_deadline(budget)
    finally:
        _ACTIVE.reset(token)
        if connection.vendor == "postgresql" and not connection.needs_rollback:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT set_config('statement_timeout', %s, %s)",
                        [str(previous), connection.in_atomic_block],
                    )
            except DatabaseError:
                # An enclosing failed transaction restores SET on rollback. Never
                # replace the original cancellation with a cleanup exception.
                if not connection.in_atomic_block:
                    connection.close()


@contextmanager
def embedding_build_transaction(budget, *, using):
    with transaction.atomic(using=using):
        with embedding_build_database(budget, using=using):
            yield
            # Check before committing sidecars/pointers, including fake-clock expiry.
            check_embedding_deadline(budget)
