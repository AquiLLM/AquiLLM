"""Real Django server-side FETCH keeps the original deadline and cursor methods."""
from time import monotonic

import pytest
from django.db import connections

from apps.collections.models import Collection
from apps.knowledge_graph.projection.database_deadline import ProjectionOperation, projection_database
from tests.integration.persisted_graph_role_support import projection_roles

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"])


def test_actual_iterator_refuses_next_fetch_after_original_deadline(projection_roles, monkeypatch):
    rows = [Collection.objects.create(name=f"projection fetch {n}") for n in range(2)]
    connection = connections["projection_source"]
    operation = ProjectionOperation(2000)
    calls, cursors = [], []
    original = connection.chunked_cursor

    def chunked_cursor():
        cursor = original()
        fetch = cursor.fetchmany

        def observed(*args, **kwargs):
            calls.append(args)
            return fetch(*args, **kwargs)

        cursor.fetchmany = observed
        cursors.append((cursor, observed))
        return cursor

    monkeypatch.setattr(connection, "chunked_cursor", chunked_cursor)
    with pytest.raises(TimeoutError, match="projection_database_deadline"):
        with projection_database("projection_source", operation=operation):
            iterator = Collection.objects.using("projection_source").filter(
                pk__in=[row.pk for row in rows]).order_by("pk").values_list("pk", flat=True).iterator(chunk_size=1)
            try:
                assert next(iterator) == rows[0].pk
                assert len(calls) == 1 and len(cursors) == 1
                operation.tighten(monotonic() - 0.001)
                next(iterator)
            finally:
                iterator.close()
    assert len(calls) == 1  # Expiry was checked before another real FETCH.
    cursor, prior = cursors[0]
    assert cursor.__dict__["fetchmany"] is prior
    assert "fetchone" not in cursor.__dict__ and "fetchall" not in cursor.__dict__


def test_nested_fetch_preserves_shorter_lock_and_statement_limits(projection_roles):
    connection = connections["projection_source"]
    operation = ProjectionOperation(2000)
    with connection.cursor() as cursor:
        original = {name: cursor.__dict__.get(name) for name in ("fetchone", "fetchmany", "fetchall")}
        with projection_database("projection_source", operation=operation):
            cursor.execute("SELECT 1")
            outer_fetch = cursor.fetchone
            assert cursor.fetchone() == (1,)
            cursor.execute("SELECT 2")
            assert cursor.fetchone is outer_fetch  # No wrapper growth on repeat execute.
            assert cursor.fetchone() == (2,)
            with connection.connection.cursor() as control:
                control.execute("SELECT set_config('lock_timeout','30ms',true),"
                                "set_config('statement_timeout','200ms',true)")
            with projection_database("projection_source", operation=operation):
                cursor.execute("SELECT 3")
                assert cursor.fetchone() == (3,)
                with connection.connection.cursor() as control:
                    control.execute("SELECT setting::bigint FROM pg_settings "
                                    "WHERE name='lock_timeout'")
                    assert 0 < control.fetchone()[0] <= 30
                    control.execute("SELECT setting::bigint FROM pg_settings "
                                    "WHERE name='statement_timeout'")
                    assert 0 < control.fetchone()[0] <= 200
            assert cursor.fetchone is outer_fetch
        assert all(cursor.__dict__.get(name) is value for name, value in original.items())
