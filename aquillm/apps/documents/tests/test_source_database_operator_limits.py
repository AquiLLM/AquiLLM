"""Actual PostgreSQL source DB scopes preserve a stricter inherited limit."""
import pytest
from django.db import connection, OperationalError, transaction

from apps.documents.services.source_loading import SourceRuntime, bounded_source_database
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


def limits():
    # Observe actual installed values without triggering an enclosing wrapper.
    with connection.wrap_database_errors, connection.connection.cursor() as cursor:
        cursor.execute("SELECT name, setting::bigint FROM pg_settings "
                       "WHERE name IN ('statement_timeout', 'lock_timeout')")
        return dict(cursor.fetchall())


def install(statement, lock):
    with connection.wrap_database_errors, connection.connection.cursor() as cursor:
        cursor.execute("SELECT set_config('statement_timeout', %s, true), "
                       "set_config('lock_timeout', %s, true)",
                       [f"{statement}ms", f"{lock}ms"])


def query():
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        assert cursor.fetchone() == (1,)


def runtime():
    # This is a DB deadline service test, not source authorization evidence.
    return SourceRuntime(TurnBudget(TurnLimits()), None)


def test_source_scope_success_keeps_operator_limit_and_restores_settings():
    with transaction.atomic():
        install(50, 13)
        before = limits()
        with bounded_source_database(runtime(), "default"):
            query()
            during = limits()
        assert limits() == before
        assert 0 < during["statement_timeout"] <= 50
        assert during["lock_timeout"] == 13
        query()


def test_shorter_operator_limit_cancels_real_source_sql_and_restores_connection():
    with transaction.atomic():
        install(10, 7)
        before = limits()
        with pytest.raises(OperationalError) as caught:
            with bounded_source_database(runtime(), "default"):
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_sleep(0.05)")
        assert caught.value.__cause__.pgcode == "57014"
        assert limits() == before
        query()


def test_nested_source_scope_keeps_limit_tightened_inside_outer_scope():
    with transaction.atomic():
        install(2000, 17)
        before = limits()
        original = runtime()
        with bounded_source_database(original, "default"):
            install(30, 17)
            tightened = limits()
            with bounded_source_database(original, "default"):
                query()
                nested = limits()
            assert limits() == tightened
            assert 0 < nested["statement_timeout"] <= 30
            assert nested["lock_timeout"] == 17
        assert limits() == before
        query()
