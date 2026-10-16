"""Native candidate composition retains one complete real PostgreSQL bound."""

import json
from contextlib import contextmanager
from dataclasses import replace

import pytest
from django.db import connection, transaction

from apps.knowledge_graph.retrieval import native_direct_seed_statement as native
from apps.knowledge_graph.retrieval.direct_seed_contracts import DirectResolutionTier
from apps.knowledge_graph.retrieval.direct_seed_sql import bounded_seed_read
from apps.knowledge_graph.retrieval.native_direct_seed_statement import candidate_rows
from apps.knowledge_graph.tests.task17_native_bound_support import (
    candidate_frame,
    observe_candidate,
    settings,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


@pytest.mark.parametrize("alias", [False, True])
def test_native_candidate_has_one_bound_and_one_control_sequence(monkeypatch, alias):
    with candidate_frame(monkeypatch) as (metadata, options):
        if alias:
            options.update(
                tier=DirectResolutionTier.ALIAS,
                lookup_field=(
                    "document_links__document_entity__mention_links__"
                    "mention__normalized_text"
                ),
            )
        budget = metadata.budget
        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout='4s'")
                cursor.execute("SET LOCAL join_collapse_limit=7")
            before = settings()
            with observe_candidate() as (counts, statements):
                rows = candidate_rows(metadata, **options)
            assert settings() == before
        assert len(statements) == 1
        assert len(rows) == 1
        assert budget.queries == 1
        assert budget.source_rows == statements[0]["rows"] == len(rows) + 1
        assert budget.wire_bytes == budget._reserved_rows == 0
        assert not budget._uncertain and not metadata.failed
        assert statements[0]["settings"][1] == ("1" if alias else "7")
        print(
            "TASK17_BOUND="
            + json.dumps(
                dict(
                    alias=alias,
                    counts=dict(counts),
                    statements=statements,
                    entity_ids=[r.entity_id for r in rows],
                    accounting=[budget.queries, budget.source_rows, budget.wire_bytes],
                )
            )
        )
        assert counts == dict(
            bounds=1,
            settings_reads=1,
            settings_writes=2,
            savepoints=1,
            releases=1,
            **({"alias_sets": 1} if alias else {}),
        )


@pytest.mark.parametrize("alias", [False, True])
def test_composition_retains_exact_sql_parameters_rows_order_and_charges(
    monkeypatch, alias
):
    with candidate_frame(monkeypatch, clock=lambda: 100.0) as (metadata, options):
        if alias:
            options.update(
                tier=DirectResolutionTier.ALIAS,
                lookup_field=(
                    "document_links__document_entity__mention_links__mention__normalized_text"
                ),
            )
        budget = metadata.budget
        with transaction.atomic(), observe_candidate() as (legacy_counts, legacy_sql):
            sql, params = native.statement(metadata, options)
            with bounded_seed_read(
                using=metadata.using,
                deadline=options["deadline"],
                clock=options["clock"],
                alias=alias,
            ):
                raw = metadata.read(sql, params, using=metadata.using, maximum=6)
            legacy = native.validate(
                raw, count=len(options["membership_states"]), limit=4
            )
        before = (budget.queries, budget.source_rows, budget.wire_bytes)
        with transaction.atomic(), observe_candidate() as (counts, actual_sql):
            actual = candidate_rows(metadata, **options)
        assert actual == legacy and len(actual) == 1
        assert actual_sql == legacy_sql
        assert (
            (
                budget.queries - before[0],
                budget.source_rows - before[1],
                budget.wire_bytes - before[2],
            )
            == before
            == (1, 2, 0)
        )
        assert legacy_counts["bounds"] == 2 and counts["bounds"] == 1


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize(
    "tighter,prior,want",
    [
        ("option", "9s", "2500ms"),
        ("budget", "9s", "5s"),
        ("option", "1200ms", "1200ms"),
    ],
)
def test_actual_candidate_settings_use_tightest_original_deadline(
    monkeypatch, alias, tighter, prior, want
):
    def clock():
        return 100.0

    with candidate_frame(monkeypatch, clock=clock) as (metadata, options):
        options["deadline"] = 102.5 if tighter == "option" else 110.0
        if alias:
            options.update(
                tier=DirectResolutionTier.ALIAS,
                lookup_field=(
                    "document_links__document_entity__mention_links__mention__normalized_text"
                ),
            )
        assert (
            metadata.runtime.clock is metadata.budget.clock is options["clock"] is clock
        )
        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)", [prior]
                )
                cursor.execute("SET LOCAL join_collapse_limit=7")
            before = settings()
            with observe_candidate() as (_, sql):
                assert len(candidate_rows(metadata, **options)) == 1
            assert sql[0]["settings"] == (want, "1" if alias else "7")
            assert settings() == before


@pytest.mark.parametrize(
    "option,value",
    [
        ("deadline", True),
        ("deadline", "100"),
        ("deadline", float("inf")),
        ("deadline", float("nan")),
        ("deadline", object()),
        ("alias", 1),
        ("alias", None),
    ],
)
def test_invalid_tightening_inputs_are_sticky_before_dispatch(
    monkeypatch, option, value
):
    with candidate_frame(monkeypatch) as (metadata, _):
        with (
            observe_candidate() as (counts, sql),
            pytest.raises((TypeError, ValueError)),
        ):
            metadata.read(
                "SELECT 1", (), using=metadata.using, maximum=1, **{option: value}
            )
        assert metadata.failed and metadata.budget.queries == 0
        assert not counts and not sql


@pytest.mark.parametrize("missing", [False, True])
def test_foreign_clock_is_rejected_without_invoking_it(monkeypatch, missing):
    def foreign():
        pytest.fail("foreign clock invoked")

    with candidate_frame(monkeypatch) as (metadata, options):
        with pytest.raises((TypeError, ValueError)):
            candidate_rows(
                metadata, **{**options, "clock": None if missing else foreign}
            )
        assert metadata.failed and metadata.budget.queries == 0


@pytest.mark.parametrize("boundary", ["cursor", "wrapper", "transaction"])
@pytest.mark.parametrize("drift", ["runtime", "owner"])
def test_owner_changes_during_cleanup_never_return_candidates(
    monkeypatch, boundary, drift
):
    from django.db.backends.utils import CursorWrapper

    from apps.collections.models import Collection

    with candidate_frame(monkeypatch) as (metadata, options):
        done, changed = [], []
        original_name = Collection.objects.get(pk=7).name

        def mutate():
            if changed or not done:
                return
            changed.append(True)
            # This actual write must roll back for inside-bound rejection.
            with connection.connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {connection.ops.quote_name(Collection._meta.db_table)} "
                    "SET name=%s WHERE id=7",
                    ["cleanup mutation"],
                )
            if drift == "runtime":
                monkeypatch.setattr(
                    metadata.runtime,
                    "clock",
                    lambda: pytest.fail("replaced clock invoked"),
                )
            else:
                monkeypatch.setattr(
                    metadata.owner, "snapshot", replace(metadata.owner.snapshot)
                )

        def observe(execute, sql, params, many, context):
            result = execute(sql, params, many, context)
            if sql.startswith("WITH expected("):
                done.append(True)
            return result

        if boundary == "cursor":
            original = CursorWrapper.__exit__

            def exit_cursor(self, *args):
                result = original(self, *args)
                mutate()
                return result

            monkeypatch.setattr(CursorWrapper, "__exit__", exit_cursor)
        elif boundary == "wrapper":
            original = connection.execute_wrapper

            @contextmanager
            def execute_wrapper(wrapper):
                with original(wrapper):
                    yield
                mutate()

            monkeypatch.setattr(connection, "execute_wrapper", execute_wrapper)
        else:
            original = transaction.Atomic.__exit__

            def exit_atomic(self, *args):
                result = original(self, *args)
                mutate()
                return result

            monkeypatch.setattr(transaction.Atomic, "__exit__", exit_atomic)
        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout='4s'")
                cursor.execute("SET LOCAL join_collapse_limit=7")
            before = settings()
            with connection.execute_wrapper(observe), pytest.raises(ValueError):
                candidate_rows(metadata, **options)
            assert settings() == before
            assert Collection.objects.get(pk=7).name == (
                "cleanup mutation" if boundary == "transaction" else original_name
            )
        assert changed and metadata.failed
        assert (
            metadata.budget.queries,
            metadata.budget.source_rows,
            metadata.budget.wire_bytes,
        ) == (1, 2, 0)


@pytest.mark.parametrize("dependency", ["meter", "bound", "bound_module"])
def test_replaced_composed_meter_dependencies_never_execute(monkeypatch, dependency):
    from apps.knowledge_graph.retrieval import direct_seed_sql, graph_seed_reads_v2

    with candidate_frame(monkeypatch) as (metadata, _):
        module, name = {
            "meter": (graph_seed_reads_v2, "_metered_seed_rows"),
            "bound": (graph_seed_reads_v2, "bounded_seed_read"),
            "bound_module": (direct_seed_sql, "bounded_seed_read"),
        }[dependency]
        monkeypatch.setattr(
            module, name, lambda *a, **kw: pytest.fail("replacement executed")
        )
        with pytest.raises(ValueError):
            metadata.read("SELECT 1", (), using=metadata.using, maximum=1)
        assert metadata.failed and metadata.budget.queries == 0
