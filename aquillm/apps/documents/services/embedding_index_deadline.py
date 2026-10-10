"""Local build transaction deadlines; no monotonic value crosses Celery."""

from contextlib import contextmanager
from django.db import connections, transaction


@contextmanager
def embedding_build_transaction(budget, *, using):
    milliseconds = int(budget.remaining_seconds() * 1000)
    if milliseconds <= 0:
        raise TimeoutError("embedding_deadline")
    with transaction.atomic(using=using):
        connection = connections[using]
        if connection.vendor == "postgresql":
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_setting('statement_timeout')")
                previous = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    [f"{milliseconds}ms"],
                )
        try:
            yield
            # Expiration during validation rolls back the sidecars/pointer too.
            if budget.remaining_seconds() <= 0:
                raise TimeoutError("embedding_deadline")
        finally:
            if connection.vendor == "postgresql" and not connection.needs_rollback:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT set_config('statement_timeout', %s, true)", [previous]
                    )
