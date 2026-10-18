"""Failure paths retain settings, nonrefundable charges and native stickiness."""

import sys
from contextlib import contextmanager
from dataclasses import replace

import pytest
from django.db import connection, transaction
from django.db.backends.utils import CursorWrapper

from apps.knowledge_graph.retrieval.direct_seed_sql import DirectSeedReadTimeout
from apps.knowledge_graph.retrieval.graph_seed_reads_v2 import _metered_seed_rows
from apps.knowledge_graph.retrieval.native_direct_seed_statement import candidate_rows
from apps.knowledge_graph.tests.task17_native_bound_support import (
    candidate_frame,
    settings,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


@pytest.mark.parametrize("deadline", [None, True, float("inf"), float("nan"), "100"])
def test_candidate_does_not_sanitize_invalid_deadline_before_validation(
    monkeypatch, deadline
):
    with candidate_frame(monkeypatch) as (metadata, options):
        with pytest.raises((TypeError, ValueError)):
            candidate_rows(metadata, **{**options, "deadline": deadline})
        assert metadata.failed and metadata.budget.queries == 0


@contextmanager
def original_settings():
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout='4s'")
            cursor.execute("SET LOCAL join_collapse_limit=7")
        before = settings()
        yield
        assert settings() == before
        with connection.cursor() as cursor:
            cursor.execute("SELECT 42")
            assert cursor.fetchone() == (42,)


@pytest.mark.parametrize(
    "failure",
    [
        "cancel",
        "python",
        "rowcount_none",
        "rowcount_negative",
        "rowcount_bool",
        "overflow",
    ],
)
def test_native_read_failure_restores_exact_settings_and_retains_work(
    monkeypatch, failure
):
    with candidate_frame(monkeypatch) as (metadata, _):
        budget = metadata.budget
        sql, maximum = "SELECT 1", 1
        expected = ValueError
        if failure == "cancel":
            sql, expected = "SELECT pg_sleep(2)", DirectSeedReadTimeout
        elif failure == "python":
            original = CursorWrapper.__getattr__

            def getattr_cursor(self, name):
                result = original(self, name)
                if name == "fetchall":

                    def fetchall():
                        result()
                        raise RuntimeError("Python fetch failure")

                    return fetchall
                return result

            monkeypatch.setattr(CursorWrapper, "__getattr__", getattr_cursor)
            expected = RuntimeError
        elif failure == "overflow":
            sql = "SELECT generate_series(1, 2)"
        else:
            original = connection.execute_wrapper

            @contextmanager
            def execute_wrapper(capture):
                def delegate(execute, sql, params, many, context):
                    def altered(sql, params, many, context):
                        result = execute(sql, params, many, context)
                        context["cursor"].rowcount = {
                            "rowcount_none": None,
                            "rowcount_negative": -1,
                            "rowcount_bool": True,
                        }[failure]
                        return result

                    return capture(altered, sql, params, many, context)

                with original(delegate):
                    yield

            monkeypatch.setattr(connection, "execute_wrapper", execute_wrapper)
        with original_settings(), pytest.raises(expected):
            metadata.read(
                sql,
                (),
                using=metadata.using,
                maximum=maximum,
                deadline=budget.clock() + (0.1 if failure == "cancel" else 2),
                clock=budget.clock,
                alias=True,
            )
        assert metadata.failed and budget.queries == 1
        assert budget.wire_bytes == 0
        if failure in {"cancel", "rowcount_none", "rowcount_negative", "rowcount_bool"}:
            assert (
                budget._uncertain
                and budget._reserved_rows == 1
                and budget.source_rows == 0
            )
        else:
            assert not budget._uncertain and budget._reserved_rows == 0
            assert budget.source_rows == (2 if failure == "overflow" else 1)
        before = (budget.queries, budget.source_rows)
        with pytest.raises((ValueError, TimeoutError)):
            metadata.read("SELECT 1", (), using=metadata.using, maximum=1)
        assert before == (budget.queries, budget.source_rows)


@pytest.mark.parametrize("dispatch", ["twice", "many", "unobserved", "mismatch"])
def test_meter_never_loses_original_physical_completeness_checks(monkeypatch, dispatch):
    with candidate_frame(monkeypatch) as (metadata, _):
        budget = metadata.budget

        def read():
            if dispatch == "unobserved":
                return ()
            with connection.cursor() as cursor:
                if dispatch == "many":
                    cursor.executemany("SELECT %s", [(1,), (2,)])
                else:
                    cursor.execute("SELECT 1")
                    if dispatch == "twice":
                        cursor.execute("SELECT 2")
                return () if dispatch == "mismatch" else tuple(cursor.fetchall())

        with original_settings(), pytest.raises(ValueError):
            _metered_seed_rows(
                read,
                using=metadata.using,
                maximum=1,
                budget=budget,
                deadline=budget.deadline,
                clock=budget.clock,
                alias=True,
            )
        assert budget.queries == 1 and budget._uncertain
        assert budget.source_rows == (1 if dispatch in {"twice", "mismatch"} else 0)
        assert budget._reserved_rows == (0 if dispatch in {"twice", "mismatch"} else 1)


@pytest.mark.parametrize("when", ["entry", "after_sql"])
def test_tighter_deadline_expiry_cannot_return_or_refund(monkeypatch, when):
    now = [100.0]
    with candidate_frame(monkeypatch, clock=lambda: now[0]) as (metadata, _):
        budget = metadata.budget

        def late(execute, sql, params, many, context):
            result = execute(sql, params, many, context)
            if sql == "SELECT 1":
                now[0] = 102.0
            return result

        if when == "entry":
            now[0] = 102.0
        with (
            original_settings(),
            connection.execute_wrapper(late),
            pytest.raises(DirectSeedReadTimeout),
        ):
            metadata.read(
                "SELECT 1",
                (),
                using=metadata.using,
                maximum=1,
                deadline=101.0,
                clock=budget.clock,
                alias=True,
            )
        assert metadata.failed
        assert (budget.queries, budget.source_rows) == (
            (0, 0) if when == "entry" else (1, 1)
        )
        assert not budget._uncertain and budget._reserved_rows == 0


def test_default_metadata_read_retains_original_bound_and_settings(monkeypatch):
    with candidate_frame(monkeypatch) as (metadata, _):
        with original_settings():
            assert metadata.read("SELECT 1", (), using=metadata.using, maximum=1) == (
                (1,),
            )
        assert not metadata.failed
        assert (metadata.budget.queries, metadata.budget.source_rows) == (1, 1)


@pytest.mark.parametrize("boundary", ["restoration", "transaction"])
def test_cleanup_exceptions_prevent_candidate_return_and_keep_charges(
    monkeypatch, boundary
):
    with candidate_frame(monkeypatch) as (metadata, options):
        failed = []

        def observe(execute, sql, params, many, context):
            result = execute(sql, params, many, context)
            if boundary == "restoration" and sql.startswith(
                "SELECT set_config('statement_timeout', %s, true),"
            ):
                failed.append(True)
                raise RuntimeError("restoration failed after actual SQL")
            return result

        if boundary == "transaction":
            original = transaction.Atomic.__exit__

            def exit_atomic(self, *args):
                result = original(self, *args)
                if metadata.budget.queries and not failed:
                    failed.append(True)
                    raise RuntimeError("transaction cleanup failed")
                return result

            monkeypatch.setattr(transaction.Atomic, "__exit__", exit_atomic)
        with (
            original_settings(),
            connection.execute_wrapper(observe),
            pytest.raises(RuntimeError),
        ):
            candidate_rows(metadata, **options)
        assert failed and metadata.failed
        assert (metadata.budget.queries, metadata.budget.source_rows) == (1, 2)


def test_final_owner_check_after_candidate_validation_is_required(monkeypatch):
    from apps.knowledge_graph.retrieval import native_direct_seed_statement as native

    with candidate_frame(monkeypatch) as (metadata, options):
        changed = []

        def profile(frame, event, arg):
            if event == "return" and frame.f_code is native.validate.__code__:
                changed.append(True)
                monkeypatch.setattr(
                    metadata.owner, "snapshot", replace(metadata.owner.snapshot)
                )

        previous = sys.getprofile()
        try:
            sys.setprofile(profile)
            with pytest.raises(ValueError):
                candidate_rows(metadata, **options)
        finally:
            sys.setprofile(previous)
        assert changed and metadata.failed
        assert (metadata.budget.queries, metadata.budget.source_rows) == (1, 2)
