"""Keep bulk provenance reads from expanding into corpus-wide nested loops."""

from functools import wraps

from django.db import connections, transaction


def canonical_provenance_scan(load_inputs):
    """Scope the PostgreSQL planner policy to this fully consumed bulk scan.

    Correlated artifact/document identity checks can be estimated as one row,
    causing PostgreSQL to repeatedly scan whole document artifacts. Preserve
    the explicit join order and favor hash/merge joins for these bounded reads.
    The SQL predicates and row-lock order remain unchanged.
    """

    @wraps(load_inputs)
    def planned_load(*args, **kwargs):
        using = kwargs["using"]
        connection = connections[using]
        if connection.vendor != "postgresql":
            return load_inputs(*args, **kwargs)

        # A savepoint restores SET LOCAL on any database or Python exception.
        # Restore explicitly on success so callers sharing the transaction do
        # not inherit a bulk-scan preference for their point lookups.
        with transaction.atomic(using=using):
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT current_setting('join_collapse_limit'), "
                    "current_setting('enable_nestloop')"
                )
                previous = cursor.fetchone()
                cursor.execute("SET LOCAL join_collapse_limit=1")
                cursor.execute("SET LOCAL enable_nestloop=off")

            result = load_inputs(*args, **kwargs)

            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT set_config('join_collapse_limit', %s, true), "
                    "set_config('enable_nestloop', %s, true)",
                    previous,
                )
            return result

    return planned_load
