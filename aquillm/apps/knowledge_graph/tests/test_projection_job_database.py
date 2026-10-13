"""Actual inherited PostgreSQL limits constrain calls without shortening jobs."""

import pytest
from django.db import DatabaseError, connection, transaction

from apps.knowledge_graph.projection import database_deadline as deadlines
from apps.knowledge_graph.tests.test_canonical_cursor_deadline_postgres import (
    raw_settings,
)

pytestmark = pytest.mark.django_db(transaction=True)


def test_inherited_statement_limit_is_not_a_permanent_job_deadline(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(deadlines, "monotonic", lambda: now[0])
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = '40ms'")
        before = raw_settings()
        with deadlines.projection_operation(300000) as job:
            with deadlines.projection_database("default") as borrowed:
                assert borrowed is job
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    assert cursor.fetchone() == (1,)
                now[0] += 6
            assert job.end() == 400.0
            assert job.remaining_ms() == 294000
        assert raw_settings() == before


@pytest.mark.parametrize("mode", ["execute", "fetch"])
def test_individual_sql_limit_fails_with_ample_job_remaining(mode):
    with deadlines.projection_operation(300000, call_timeout_ms=30) as job:
        with pytest.raises(DatabaseError, match="statement timeout"):
            with deadlines.projection_database("default"):
                factory = (
                    connection.cursor
                    if mode == "execute"
                    else connection.chunked_cursor
                )
                with factory() as cursor:
                    cursor.execute("SELECT pg_sleep(0.15)")
                    cursor.fetchone()
        assert job.remaining_ms() > 290000


def test_multiple_successful_calls_share_original_job_and_restore_settings(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(deadlines, "monotonic", lambda: now[0])
    with transaction.atomic():
        before = raw_settings()
        observed = []

        def inspect(execute, sql, params, many, context):
            observed.append(raw_settings()["statement_timeout"])
            return execute(sql, params, many, context)

        with deadlines.projection_operation(300000, call_timeout_ms=5000) as job:
            with (
                deadlines.projection_database("default"),
                connection.execute_wrapper(inspect),
            ):
                with connection.cursor() as cursor:
                    for _ in range(3):
                        cursor.execute("SELECT 1")
                        assert cursor.fetchone() == (1,)
                        assert raw_settings() == before
                        now[0] += 4
                assert job.end() == 400.0
            assert raw_settings() == before
        assert observed and all(0 < value <= 5000 for value in observed)


def test_cursor_result_is_rejected_if_per_call_time_expires(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(deadlines, "monotonic", lambda: now[0])
    with deadlines.projection_operation(300000, call_timeout_ms=5000) as job:
        with pytest.raises(TimeoutError):
            with deadlines.projection_database("default"):
                with connection.cursor() as cursor:
                    original = cursor.cursor

                    class Delayed:
                        def __getattr__(self, name):
                            return getattr(original, name)

                        def execute(self, *args, **kwargs):
                            result = original.execute(*args, **kwargs)
                            now[0] += 5.1
                            return result

                    cursor.cursor = Delayed()
                    cursor.execute("SELECT 1")
        assert job.remaining_ms() > 290000


def test_explicit_ambient_tightening_survives_independent_call_restoration():
    with deadlines.projection_operation(300000, call_timeout_ms=5000):
        with deadlines.projection_database("default"):
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout = '40ms'")
                assert raw_settings()["statement_timeout"] == 40
                cursor.execute("SELECT 1")
                assert cursor.fetchone() == (1,)
                assert raw_settings()["statement_timeout"] == 40


def test_borrowed_call_deadline_cannot_be_restarted_by_reentrant_cursor(monkeypatch):
    now, calls = [100.0], []
    monkeypatch.setattr(deadlines, "monotonic", lambda: now[0])

    def reenter(execute, sql, params, many, context):
        if not calls:
            calls.append(sql)
            now[0] += 5.1
            with connection.cursor() as nested:
                nested.execute("SELECT 2")
        return execute(sql, params, many, context)

    with deadlines.projection_operation(300000, call_timeout_ms=5000) as job:
        with pytest.raises(TimeoutError):
            with (
                deadlines.projection_database("default"),
                connection.execute_wrapper(reenter),
            ):
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
        assert job.remaining_ms() > 290000
    assert calls == ["SELECT 1"]


def test_short_nested_call_does_not_retroactively_expire_its_parent(monkeypatch):
    now, calls = [100.0], []
    monkeypatch.setattr(deadlines, "monotonic", lambda: now[0])

    def reenter(execute, sql, params, many, context):
        if not calls:
            calls.append(sql)
            now[0] += 1
            with connection.connection.cursor() as control:
                control.execute("SET LOCAL statement_timeout = '40ms'")
            with connection.cursor() as nested:
                nested.execute("SELECT 2")
                assert nested.fetchone() == (2,)
        return execute(sql, params, many, context)

    with deadlines.projection_operation(300000, call_timeout_ms=5000) as job:
        with (
            deadlines.projection_database("default"),
            connection.execute_wrapper(reenter),
        ):
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                assert cursor.fetchone() == (1,)
        assert job.end() == 400.0
