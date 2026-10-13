"""Actual restricted connections preserve finite and inherited PostgreSQL bounds."""
from concurrent.futures import ThreadPoolExecutor
from time import monotonic

import pytest
from django.db import OperationalError, connections

from apps.knowledge_graph.projection.database_deadline import (
    ProjectionOperation, projection_database, projection_operation,
)
from tests.integration.persisted_graph_role_support import projection_roles

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"])


def settings_for(alias):
    with connections[alias].cursor() as cursor:
        cursor.execute("SELECT name,setting::bigint FROM pg_settings "
                       "WHERE name IN ('lock_timeout','statement_timeout')")
        return dict(cursor.fetchall())


@pytest.mark.parametrize("cancelled", [False, True])
def test_shorter_inherited_limits_restore_after_actual_sql(projection_roles, cancelled):
    alias = "projection_source"
    original = settings_for(alias)
    try:
        with connections[alias].cursor() as cursor:
            cursor.execute("SELECT set_config('statement_timeout','100ms',false),"
                           "set_config('lock_timeout','50ms',false)")
        expected = settings_for(alias)
        operation = ProjectionOperation(2000)
        started = monotonic()
        if cancelled:
            with pytest.raises(OperationalError, match="statement timeout"):
                with projection_database(alias, operation=operation):
                    with connections[alias].cursor() as cursor:
                        cursor.execute("SELECT pg_sleep(0.5)")
        else:
            with projection_database(alias, operation=operation):
                observed = settings_for(alias)
                assert 0 < observed["statement_timeout"] <= 100
                assert 0 < observed["lock_timeout"] <= 50
        assert monotonic() - started < 1.5
        assert settings_for(alias) == expected
        assert operation.end() <= started + 0.15
    finally:
        with connections[alias].cursor() as cursor:
            cursor.execute("SELECT set_config('statement_timeout',%s,false),"
                           "set_config('lock_timeout',%s,false)",
                           [str(original["statement_timeout"]), str(original["lock_timeout"])])


def test_nested_alias_scopes_share_original_operation(projection_roles):
    operation = ProjectionOperation(2000)
    with projection_operation(operation=operation):
        with projection_database("projection_source", operation=operation):
            original_end = operation.end()
            with projection_database("projection_state", operation=operation):
                assert operation.end() <= original_end
                assert operation.remaining_ms() <= 2000
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(operation.tighten, [original_end - 0.25, original_end - 0.5]))
    assert operation.end() <= original_end - 0.5

