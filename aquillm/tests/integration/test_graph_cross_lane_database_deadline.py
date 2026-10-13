"""Real canonical/projection nesting shares each original PostgreSQL bound."""
from contextlib import ExitStack
from time import monotonic

import pytest
from django.db import connection

from apps.knowledge_graph.projection.database_deadline import ProjectionOperation, projection_database
from apps.knowledge_graph.resolution.canonical_deadline_v2 import bounded_transaction

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("outer", ["canonical", "projection"])
def test_canonical_projection_nested_cursor_keeps_original_minima(outer):
    connection.ensure_connection()
    with connection.connection.cursor() as raw:
        raw.execute("SELECT name, setting::bigint FROM pg_settings "
                    "WHERE name IN ('lock_timeout','statement_timeout')")
        original = dict(raw.fetchall())
    operation = ProjectionOperation(2000)
    now = monotonic()
    clock = [now]
    contexts = {
        "canonical": bounded_transaction(using="default", deadline=now + 1, clock=lambda: clock[0]),
        "projection": projection_database("default", operation=operation),
    }
    with ExitStack() as stack:
        stack.enter_context(contexts[outer])
        stack.enter_context(contexts["projection" if outer == "canonical" else "canonical"])
        clock[0] = now + 0.8  # No fresh allowance in the nested operation.
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 UNION ALL SELECT 2")
            assert cursor.fetchone() == (1,)
            clock[0] = now + 0.9
            assert cursor.fetchone() == (2,)
            with connection.connection.cursor() as raw:
                raw.execute("SELECT name, setting::bigint FROM pg_settings "
                            "WHERE name IN ('lock_timeout','statement_timeout')")
                current = dict(raw.fetchall())
            assert all(0 < current[name] <= 100 for name in current)
            assert operation.remaining_ms() > 100
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        assert cursor.fetchone() == (1,)
        cursor.execute("SELECT name, setting::bigint FROM pg_settings "
                       "WHERE name IN ('lock_timeout','statement_timeout')")
        assert dict(cursor.fetchall()) == original
