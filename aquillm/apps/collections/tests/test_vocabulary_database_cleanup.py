"""Exact savepoint cleanup remains possible without admitting expired body SQL."""
import pytest
from django.db import OperationalError, connection, transaction

from apps.collections.services import vocabulary_database as database
from apps.collections.tests.test_vocabulary_database_operator_limits import install, limits, query

pytestmark = pytest.mark.django_db(transaction=True)


def test_expired_nested_cancellation_rolls_back_without_replacing_original_error(monkeypatch):
    observed, cleanup, outer_deadline = [], [], []
    real_clock = database.monotonic
    expired = []

    def clock():
        return expired[0] if expired else real_clock()

    def observe(execute, sql, params, many, context):
        if sql.startswith(("ROLLBACK TO SAVEPOINT", "RELEASE SAVEPOINT")):
            cleanup.append(sql)
        try:
            return execute(sql, params, many, context)
        except OperationalError as error:
            if sql == "SELECT pg_sleep(0.15)":
                observed.append(error)
                expired.append(outer_deadline[0] + 1)
            raise

    monkeypatch.setattr(database, "monotonic", clock)
    with transaction.atomic():
        install(2000, 1000)
        before = limits()
        original_connection = connection.connection
        with connection.execute_wrapper(observe), pytest.raises(OperationalError) as caught:
            with database.vocabulary_database("default"):
                outer_deadline.append(database._DEADLINE.get())
                install(30, 17)
                with database.vocabulary_database("default"):
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT pg_sleep(0.15)")
        assert observed == [caught.value]
        assert caught.value.__cause__.pgcode == "57014"
        assert sum(sql.startswith("ROLLBACK TO SAVEPOINT") for sql in cleanup) == 2
        assert sum(sql.startswith("RELEASE SAVEPOINT") for sql in cleanup) == 2
        assert connection.connection is original_connection
        assert not connection.needs_rollback
        assert database._DEADLINE.get() is None and database._CLEANUP.get() is None
        assert limits() == before
        query()


@pytest.mark.parametrize("statement", ["body", "rollback", "release", "savepoint"])
def test_expired_body_cannot_use_cleanup_shaped_sql_to_bypass_deadline(monkeypatch, statement):
    with transaction.atomic():
        install(2000, 1000)
        before = limits()
        with database.vocabulary_database("default"):
            original_deadline = database._DEADLINE.get()
            sid = connection.savepoint_ids[-1]
            sql = {"body": "SELECT 1", "rollback": connection.ops.savepoint_rollback_sql(sid),
                   "release": connection.ops.savepoint_commit_sql(sid),
                   "savepoint": connection.ops.savepoint_create_sql("body_probe")}[statement]
            calls = []

            def observe(execute, query_sql, params, many, context):
                calls.append(query_sql)
                return execute(query_sql, params, many, context)

            with monkeypatch.context() as clock:
                clock.setattr(database, "monotonic", lambda: original_deadline + 1)
                with connection.execute_wrapper(observe), pytest.raises(ValueError, match="deadline"):
                    with connection.cursor() as cursor:
                        cursor.execute(sql)
            assert calls == [] and database._CLEANUP.get() is None
            assert database._DEADLINE.get() == original_deadline
            query()
        assert limits() == before
        assert database._DEADLINE.get() is None and database._CLEANUP.get() is None
        query()
