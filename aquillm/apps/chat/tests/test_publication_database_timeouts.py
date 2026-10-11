"""Real PostgreSQL publication work inherits stricter connection limits."""
from contextlib import contextmanager

import pytest
from django.db import connection, DataError, OperationalError, transaction

from apps.chat.services import rag_publication_database as publication
from lib.llm.answer_publication import AnswerPublicationRejected

pytestmark = pytest.mark.django_db(transaction=True)


def settings():
    with connection.cursor() as cursor:
        cursor.execute("SELECT name, setting::bigint FROM pg_settings "
                       "WHERE name IN ('statement_timeout', 'lock_timeout')")
        return dict(cursor.fetchall())


def raw_settings():
    # Inspect what the preceding real query installed without another wrapper pass.
    with connection.wrap_database_errors, connection.connection.cursor() as cursor:
        cursor.execute("SELECT name, setting::bigint FROM pg_settings "
                       "WHERE name IN ('statement_timeout', 'lock_timeout')")
        return dict(cursor.fetchall())


def install(statement, lock):
    with connection.cursor() as cursor:
        cursor.execute("SELECT set_config('statement_timeout', %s, true), "
                       "set_config('lock_timeout', %s, true)",
                       [f"{statement}ms", f"{lock}ms"])


def query():
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        assert cursor.fetchone() == (1,)


def test_shorter_operator_statement_timeout_cancels_real_publication_sql():
    with transaction.atomic():
        install(10, 7)
        before = settings()
        with pytest.raises(OperationalError):
            with publication._bounded_database("default", 1):
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_sleep(0.05)")
        assert settings() == before
        query()


@pytest.mark.parametrize("outer,inner", [(0.8, 0.08), (0.08, 0.8)])
def test_nested_scopes_keep_shortest_active_limit_and_restore(outer, inner):
    with transaction.atomic():
        install(2000, 17)
        before = settings()
        with publication._bounded_database("default", outer):
            query()
            outer_limit = raw_settings()
            assert 0 < outer_limit["statement_timeout"] <= int(outer * 1000)
            with publication._bounded_database("default", inner):
                query()
                nested = raw_settings()
                assert 0 < nested["statement_timeout"] <= int(min(outer, inner) * 1000)
                assert nested["lock_timeout"] == 17
            query()
            restored_outer = raw_settings()
            assert 0 < restored_outer["statement_timeout"] <= int(outer * 1000)
            assert restored_outer["lock_timeout"] == 17
        assert settings() == before


def test_success_preserves_operator_minimum_and_restores_exact_settings():
    with transaction.atomic():
        install(50, 13)
        before = settings()
        with publication._bounded_database("default", 1):
            query()
            during = raw_settings()
            assert 0 < during["statement_timeout"] <= 50
            assert during["lock_timeout"] == 13
        assert settings() == before


def test_inner_entry_preserves_operator_limit_lowered_inside_outer_scope():
    with transaction.atomic():
        install(2000, 17)
        before = settings()
        with publication._bounded_database("default", 1):
            query()
            # Use real control SQL so the outer wrapper cannot replace the value
            # during this test's observation of the operator's stricter choice.
            with connection.wrap_database_errors, connection.connection.cursor() as cursor:
                cursor.execute("SELECT set_config('statement_timeout', '30ms', true)")
            assert raw_settings()["statement_timeout"] == 30
            with publication._bounded_database("default", 0.8):
                query()
                assert 0 < raw_settings()["statement_timeout"] <= 30
            assert raw_settings()["statement_timeout"] == 30
        assert settings() == before


def test_setup_time_is_inside_original_absolute_deadline(monkeypatch):
    with transaction.atomic():
        install(0, 0)
        before = settings()
        elapsed = [0.0]
        real = publication.monotonic
        start = real()
        monkeypatch.setattr(publication, "monotonic", lambda: start + elapsed[0])
        original_read = publication._read_limits
        observed = []
        def setup_delay(conn):
            result = original_read(conn)
            observed.append(result)
            elapsed[0] = 1.0
            return result
        monkeypatch.setattr(publication, "_read_limits", setup_delay)
        with pytest.raises(AnswerPublicationRejected):
            with publication._bounded_database("default", 0.5):
                query()
        assert observed == [before]
        assert settings() == before


def test_nested_statement_error_rolls_back_and_keeps_outer_connection_usable():
    with transaction.atomic():
        install(2000, 17)
        before = settings()
        with publication._bounded_database("default", 1):
            with connection.wrap_database_errors, connection.connection.cursor() as cursor:
                cursor.execute("SELECT set_config('statement_timeout', '10ms', true)")
            with pytest.raises(OperationalError):
                with publication._bounded_database("default", 0.8):
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT pg_sleep(0.05)")
            query()
            assert 0 < raw_settings()["statement_timeout"] <= 10
        assert settings() == before
        query()


def test_expired_outer_scope_does_not_prevent_nested_rollback(monkeypatch):
    with transaction.atomic():
        install(2000, 17)
        before = settings()
        start = publication.monotonic()
        elapsed = [0.0]
        monkeypatch.setattr(publication, "monotonic", lambda: start + elapsed[0])
        with pytest.raises(RuntimeError, match="original operation failure"):
            with publication._bounded_database("default", 0.5):
                with publication._bounded_database("default", 0.4):
                    query()
                    elapsed[0] = 1.0
                    raise RuntimeError("original operation failure")
        assert settings() == before
        query()


def test_failed_real_restoration_discards_connection(monkeypatch):
    original = publication._control_cursor
    restored = []
    @contextmanager
    def fail_restore(conn):
        with original(conn) as cursor:
            class Cursor:
                def execute(self, sql, params=None):
                    if sql.startswith("SELECT set_config('statement_timeout', %s"):
                        restored.append(True)
                        cursor.execute("SELECT 1 / 0")
                    return cursor.execute(sql, params)
                def fetchall(self):
                    return cursor.fetchall()
            yield Cursor()
    monkeypatch.setattr(publication, "_control_cursor", fail_restore)
    with pytest.raises(DataError):
        with publication._bounded_database("default", 1):
            query()
    assert restored == [True]
    assert connection.connection is None
    monkeypatch.setattr(publication, "_control_cursor", original)
    query()
