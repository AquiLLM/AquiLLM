"""Actual vocabulary queries preserve later PostgreSQL operator minima."""
import pytest
from django.db import OperationalError, connection, transaction

from apps.collections.services import vocabulary_database as database

pytestmark = pytest.mark.django_db(transaction=True)


def limits():
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


@pytest.mark.parametrize("later", [None, "statement", "lock", "both"])
def test_vocabulary_data_query_retains_operator_minima(later):
    with transaction.atomic():
        install(30 if later is None else 2000, 17 if later is None else 1000)
        before = limits()
        with database.vocabulary_database("default"):
            deadline = database._DEADLINE.get()
            if later is not None:
                install(30 if later in ("statement", "both") else 2000,
                        17 if later in ("lock", "both") else 1000)
            required = limits()
            query()
            during = limits()
            assert 0 < during["statement_timeout"] <= required["statement_timeout"]
            assert 0 < during["lock_timeout"] <= required["lock_timeout"]
            assert database._DEADLINE.get() == deadline
        assert limits() == before
        assert database._DEADLINE.get() is None
        query()


def test_nested_vocabulary_savepoint_preserves_new_operator_limits():
    with transaction.atomic():
        install(2000, 1000)
        before = limits()
        with database.vocabulary_database("default"):
            deadline = database._DEADLINE.get()
            install(30, 17)
            tightened = limits()
            with database.vocabulary_database("default"):
                query()
                assert 0 < limits()["statement_timeout"] <= 30
                assert 0 < limits()["lock_timeout"] <= 17
                assert database._DEADLINE.get() <= deadline
            assert database._DEADLINE.get() == deadline
            assert limits() == tightened
        assert limits() == before
        query()


def test_vocabulary_operator_cancellation_preserves_error_and_outer_connection():
    with transaction.atomic():
        install(2000, 1000)
        before = limits()
        with database.vocabulary_database("default"):
            deadline = database._DEADLINE.get()
            install(30, 17)
            with pytest.raises(OperationalError) as caught:
                with database.vocabulary_database("default"):
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT pg_sleep(0.15)")
            assert caught.value.__cause__.pgcode == "57014"
            assert limits() == {"statement_timeout": 30, "lock_timeout": 17}
            assert database._DEADLINE.get() == deadline
            query()
            assert 0 < limits()["statement_timeout"] <= 30
        assert limits() == before
        assert database._DEADLINE.get() is None
        query()
