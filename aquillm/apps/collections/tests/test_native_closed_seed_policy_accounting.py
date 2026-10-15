"""Focused native closed-seed policy contracts."""

from contextlib import contextmanager

import pytest
from django.db import connection

from apps.documents.models import RawTextDocument
from apps.knowledge_graph.retrieval.extended_seed_repository import (
    ExtendedSeedRepository,
)

from .native_closed_seed_policy_fixtures import (
    native_frame,
    prepare,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


def test_native_final_policy_reservation_uses_one_unspent_row_and_keeps_sentinel(
    monkeypatch,
):
    with native_frame(monkeypatch, max_source_rows=52) as frame:
        prepare(frame)
        budget = frame[4]
        assert frame[1].mapping
        assert (budget.queries, budget.source_rows, budget._reserved_rows) == (6, 51, 0)
        assert budget.max_source_rows == 52


@pytest.mark.parametrize(
    "bad_rows", ["missing", "extra", "boolean", "negative", "short", "wrong"]
)
def test_malformed_actual_proof_results_never_authorize_or_retry(monkeypatch, bad_rows):
    statements = {
        "missing": "SELECT 2,2,2,2,2 WHERE FALSE",
        "extra": "SELECT 2,2,2,2,2 UNION ALL SELECT 2,2,2,2,2",
        "boolean": "SELECT TRUE,2,2,2,2",
        "negative": "SELECT -1,2,2,2,2",
        "short": "SELECT 2,2,2,2",
        "wrong": "SELECT 2,2,2,1,2",
    }
    with native_frame(monkeypatch) as frame:
        calls = []

        def corrupt(execute, sql, params, many, context):
            if sql.startswith("WITH native_seed_policy_permissions"):
                calls.append(True)
                return execute(statements[bad_rows], None, many, context)
            return execute(sql, params, many, context)

        with connection.execute_wrapper(corrupt), pytest.raises(ValueError):
            prepare(frame)
        assert calls == [True] and frame[1].mapping is None
        spent = (frame[4].queries, frame[4].source_rows)
        with (
            connection.execute_wrapper(corrupt),
            pytest.raises((ValueError, TimeoutError)),
        ):
            prepare(frame)
        assert calls == [True] and spent == (frame[4].queries, frame[4].source_rows)


@pytest.mark.parametrize(
    "change",
    [
        "principal",
        "method",
        "field",
        "settings",
        "late",
        "unknown",
        "manager",
        "user_field",
        "repository",
    ],
)
def test_mid_statement_changes_abort_after_actual_work_without_fallback(
    monkeypatch, change
):
    from asyncio import CancelledError

    with native_frame(monkeypatch) as frame:
        active, owner, _, _, budget = frame
        authorization = active[3].authorization
        calls = []

        def mutate(execute, sql, params, many, context):
            native = sql.startswith("WITH native_seed_policy_permissions")
            if native:
                calls.append(True)
            if native and change == "unknown":

                class Unknown:
                    rowcount = -1

                context["cursor"] = Unknown()
            result = execute(sql, params, many, context)
            if native:
                capability = authorization.reauthorization_capability
                if change == "principal":
                    from copy import copy

                    object.__setattr__(
                        capability, "_principal", copy(capability._principal)
                    )
                elif change == "method":
                    monkeypatch.setattr(
                        capability._policy,
                        "_principal_id",
                        lambda _: capability._principal.pk,
                    )
                elif change == "field":
                    monkeypatch.setattr(
                        RawTextDocument._meta.get_field("id"), "column", "replaced_id"
                    )
                elif change == "settings":
                    active[5].settings.graph_extended_max_seeds += 1
                elif change == "late":
                    active[2].budget.close("test-late-proof")
                elif change == "unknown":
                    pass
                elif change == "manager":
                    manager = RawTextDocument._base_manager
                    original = manager.get_queryset
                    monkeypatch.setitem(
                        manager.__dict__, "get_queryset", lambda: original()
                    )
                elif change == "repository":
                    monkeypatch.setattr(
                        ExtendedSeedRepository,
                        "_load_seed_identities_batch",
                        lambda self, **kw: {},
                    )
                else:
                    from django.contrib.auth.models import User

                    monkeypatch.setattr(
                        User._meta.get_field("is_active"), "column", "replaced_active"
                    )
            return result

        with (
            connection.execute_wrapper(mutate),
            pytest.raises((ValueError, TimeoutError, CancelledError)),
        ):
            prepare(frame)
        assert calls == [True] and owner.mapping is None and budget.queries == 1
        assert budget.source_rows == (0 if change == "unknown" else 1)
        assert budget._uncertain is (change == "unknown")


def test_actual_postgres_policy_timeout_poisons_original_ledger_without_retry(
    monkeypatch,
):
    from apps.knowledge_graph.retrieval.direct_seed_sql import DirectSeedReadTimeout

    with native_frame(monkeypatch, frontier_timeout=0.2) as frame:
        calls = []

        def delayed(execute, sql, params, many, context):
            if sql.startswith("WITH native_seed_policy_permissions"):
                calls.append(True)
                return execute("SELECT pg_sleep(1),2,2,2,2", None, many, context)
            return execute(sql, params, many, context)

        with connection.execute_wrapper(delayed), pytest.raises(DirectSeedReadTimeout):
            prepare(frame)
        budget = frame[4]
        assert calls == [True] and budget.queries == 1 and budget.source_rows == 0
        assert (
            budget._uncertain
            and budget._reserved_rows == 1
            and frame[1].mapping is None
        )
        with (
            connection.execute_wrapper(delayed),
            pytest.raises((ValueError, TimeoutError)),
        ):
            prepare(frame)
        assert calls == [True] and budget.queries == 1


def test_mutation_during_transaction_cleanup_is_rejected_after_charging(monkeypatch):
    from apps.knowledge_graph.retrieval import graph_seed_reads_v2 as reads

    with native_frame(monkeypatch) as frame:
        original = reads.bounded_seed_read

        @contextmanager
        def cleanup(**kwargs):
            with original(**kwargs):
                yield
            monkeypatch.setattr(
                RawTextDocument._meta.get_field("id"), "column", "cleanup_replacement"
            )

        monkeypatch.setattr(reads, "bounded_seed_read", cleanup)
        with pytest.raises(ValueError):
            prepare(frame)
        assert frame[4].queries == frame[4].source_rows == 1
        assert frame[1].mapping is None
