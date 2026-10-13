"""The original reconciliation deadline bounds every real PostgreSQL FETCH."""
import pytest
from django.db import DatabaseError, connection, transaction

from apps.collections.models import Collection
from apps.knowledge_graph.resolution.canonical_deadline_v2 import (
    ReconciliationDeadlineExceeded,
    bounded_transaction,
)

pytestmark = pytest.mark.django_db(transaction=True)


class ObservedDriver:
    """Delegate to the real driver, recording attempts below Django's guards."""

    def __init__(self, cursor, calls, after=None):
        self.cursor, self.calls = cursor, calls
        self.after = after

    def __getattr__(self, name):
        return getattr(self.cursor, name)

    def fetchone(self, *args):
        self.calls.append("fetchone")
        result = self.cursor.fetchone(*args)
        if self.after is not None:
            self.after()
        return result

    def fetchmany(self, *args):
        self.calls.append("fetchmany")
        return self.cursor.fetchmany(*args)

    def fetchall(self, *args):
        self.calls.append("fetchall")
        return self.cursor.fetchall(*args)


def raw_settings():
    with connection.connection.cursor() as cursor:
        cursor.execute("SELECT name, setting::bigint FROM pg_settings "
                       "WHERE name IN ('lock_timeout','statement_timeout')")
        return dict(cursor.fetchall())


@pytest.mark.parametrize("method", ["fetchone", "fetchmany", "fetchall"])
def test_named_cursor_refuses_expired_fetch_before_driver_call(method):
    now, calls = [100.0], []
    cursor = None
    with pytest.raises(ReconciliationDeadlineExceeded):
        with bounded_transaction(using="default", deadline=103.0, clock=lambda: now[0]):
            with connection.chunked_cursor() as cursor:
                cursor.cursor = ObservedDriver(cursor.cursor, calls)
                cursor.execute("SELECT generate_series(1,3)")
                assert cursor.fetchone() == (1,)
                now[0] = 104.0
                getattr(cursor, method)()
    assert calls == ["fetchone"]
    assert all(name not in cursor.__dict__ for name in ("fetchone", "fetchmany", "fetchall"))


def test_queryset_iterator_stops_before_expired_next_page(monkeypatch):
    rows = [Collection.objects.create(name=f"canonical page {n}") for n in range(2)]
    now, calls, cursors = [100.0], [], []
    original = connection.chunked_cursor

    def observed_cursor():
        cursor = original()
        cursor.cursor = ObservedDriver(cursor.cursor, calls)
        cursors.append(cursor)
        return cursor

    monkeypatch.setattr(connection, "chunked_cursor", observed_cursor)
    with pytest.raises(ReconciliationDeadlineExceeded):
        with bounded_transaction(using="default", deadline=103.0, clock=lambda: now[0]):
            iterator = Collection.objects.filter(pk__in=[row.pk for row in rows]).order_by(
                "pk").values_list("pk", flat=True).iterator(chunk_size=1)
            try:
                assert next(iterator) == rows[0].pk
                now[0] = 104.0
                next(iterator)
            finally:
                iterator.close()
    assert len(cursors) == 1 and calls == ["fetchmany"]


def test_success_restores_limits_inside_enclosing_atomic():
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL lock_timeout = '0'")
            cursor.execute("SET LOCAL statement_timeout = '0'")
        before = raw_settings()
        with bounded_transaction(using="default", deadline=103.0, clock=lambda: 100.0):
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                assert cursor.fetchone() == (1,)
                assert raw_settings() == {"lock_timeout": 3000, "statement_timeout": 3000}
        assert raw_settings() == before


def test_nested_shorter_deadline_cannot_be_overwritten_by_outer_execute():
    with bounded_transaction(using="default", deadline=110.0, clock=lambda: 100.0):
        # Simulate a nested operation with no additional operator restriction.
        with connection.connection.cursor() as control:
            control.execute("SELECT set_config('lock_timeout','0',true),"
                            "set_config('statement_timeout','0',true)")
        with bounded_transaction(using="default", deadline=101.0, clock=lambda: 100.0):
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                assert cursor.fetchone() == (1,)
                assert raw_settings() == {"lock_timeout": 1000, "statement_timeout": 1000}


def test_successful_pages_and_existing_fetch_attribute_are_preserved():
    calls = []
    with transaction.atomic(), connection.chunked_cursor() as cursor:
        cursor.cursor = ObservedDriver(cursor.cursor, calls)
        prior = cursor.fetchmany
        cursor.fetchmany = prior
        with bounded_transaction(using="default", deadline=103.0, clock=lambda: 100.0):
            cursor.execute("SELECT generate_series(1,3)")
            assert cursor.fetchmany(1) == [(1,)]
            assert cursor.fetchmany(1) == [(2,)]
            assert cursor.fetchall() == [(3,)]
        assert calls == ["fetchmany", "fetchmany", "fetchall"]
        assert cursor.fetchmany is prior
        assert "fetchone" not in cursor.__dict__ and "fetchall" not in cursor.__dict__


def test_inner_deadline_also_bounds_outer_created_named_cursor_fetch():
    with bounded_transaction(using="default", deadline=110.0, clock=lambda: 100.0):
        with connection.chunked_cursor() as cursor:
            cursor.execute("SELECT generate_series(1,3)")
            assert cursor.fetchone() == (1,)
            with bounded_transaction(using="default", deadline=101.0, clock=lambda: 100.0):
                assert cursor.fetchone() == (2,)
                assert raw_settings() == {"lock_timeout": 1000, "statement_timeout": 1000}
            assert cursor.fetchone() == (3,)
            assert raw_settings() == {"lock_timeout": 10000, "statement_timeout": 10000}


def test_caught_inner_error_restores_surviving_outer_limits():
    with bounded_transaction(using="default", deadline=110.0, clock=lambda: 100.0):
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            assert cursor.fetchone() == (1,)
            before = raw_settings()
            with pytest.raises(ValueError, match="inner failure"):
                with bounded_transaction(using="default", deadline=101.0, clock=lambda: 100.0):
                    cursor.execute("SELECT 2")
                    assert cursor.fetchone() == (2,)
                    raise ValueError("inner failure")
            assert raw_settings() == before
            cursor.execute("SELECT 3")
            assert cursor.fetchone() == (3,)
            assert raw_settings() == before


@pytest.mark.parametrize("expiring", ["outer", "inner"])
def test_fetch_discards_result_if_either_enclosing_deadline_expires(expiring):
    now, returned, calls = {"outer": 100.0, "inner": 100.0}, [], []

    def after_fetch():
        now[expiring] = 104.0

    with pytest.raises(ReconciliationDeadlineExceeded):
        with bounded_transaction(using="default", deadline=103.0, clock=lambda: now["outer"]):
            with connection.chunked_cursor() as cursor:
                cursor.cursor = ObservedDriver(cursor.cursor, calls, after=after_fetch)
                cursor.execute("SELECT 1")
                with bounded_transaction(using="default", deadline=103.0, clock=lambda: now["inner"]):
                    returned.append(cursor.fetchone())
    assert calls == ["fetchone"] and returned == []


def test_operator_statement_limit_on_fetch_retains_django_error_type():
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = '40ms'")
        with pytest.raises(DatabaseError, match="statement timeout"):
            with bounded_transaction(using="default", deadline=103.0, clock=lambda: 100.0):
                with connection.chunked_cursor() as cursor:
                    cursor.execute("SELECT pg_sleep(0.2)")
                    cursor.fetchone()
        assert raw_settings()["statement_timeout"] == 40


def test_ambient_tightening_survives_nested_savepoint_execute_and_fetch():
    with bounded_transaction(using="default", deadline=110.0, clock=lambda: 100.0):
        with connection.connection.cursor() as control:
            control.execute("SELECT set_config('lock_timeout','30ms',true),"
                            "set_config('statement_timeout','200ms',true)")
        with bounded_transaction(using="default", deadline=105.0, clock=lambda: 100.0):
            with connection.chunked_cursor() as cursor:
                cursor.execute("SELECT generate_series(1,2)")
                assert cursor.fetchone() == (1,)
                assert raw_settings() == {"lock_timeout": 30, "statement_timeout": 200}
                assert cursor.fetchone() == (2,)
        assert raw_settings() == {"lock_timeout": 30, "statement_timeout": 200}
