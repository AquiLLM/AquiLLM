"""Actual PostgreSQL build scopes retain later, stricter operator timeouts."""
import pytest
from django.db import connection, OperationalError, transaction

from apps.documents.services.embedding_index_deadline import (
    embedding_build_database, embedding_build_transaction,
)
from lib.embeddings.document_contracts import build_budget

pytestmark = pytest.mark.django_db(transaction=True)


def limits():
    # Raw driver observation avoids accidentally invoking the scope under test.
    with connection.wrap_database_errors, connection.connection.cursor() as cursor:
        cursor.execute("SELECT name, setting::bigint FROM pg_settings "
                       "WHERE name IN ('statement_timeout', 'lock_timeout')")
        return dict(cursor.fetchall())


def install(statement):
    with connection.wrap_database_errors, connection.connection.cursor() as cursor:
        cursor.execute("SELECT set_config('statement_timeout', %s, true)",
                       [f"{statement}ms"])


def query():
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        assert cursor.fetchone() == (1,)


@pytest.mark.parametrize("tighten_after_entry", [False, True])
def test_build_data_statement_preserves_actual_operator_minimum(tighten_after_entry):
    with transaction.atomic():
        install(2000 if tighten_after_entry else 30)
        before = limits()
        budget = build_budget()
        deadline = budget.deadline_at
        with embedding_build_database(budget, using="default"):
            if tighten_after_entry:
                install(30)
            assert limits()["statement_timeout"] == 30
            query()
            during = limits()
            assert 0 < during["statement_timeout"] <= 30
            assert during["lock_timeout"] == before["lock_timeout"]
            assert budget.deadline_at == deadline
        assert limits() == before
        query()


def test_nested_build_transaction_keeps_later_tighter_operator_limit():
    with transaction.atomic():
        install(2000)
        before = limits()
        budget = build_budget()
        with embedding_build_database(budget, using="default"):
            install(30)
            tightened = limits()
            with embedding_build_transaction(budget, using="default"):
                query()
                assert 0 < limits()["statement_timeout"] <= 30
            assert limits() == tightened
        assert limits() == before
        query()


def test_later_operator_timeout_keeps_original_error_and_usable_connection():
    with transaction.atomic():
        install(2000)
        before = limits()
        budget = build_budget()
        with embedding_build_database(budget, using="default"):
            install(30)
            with pytest.raises(OperationalError) as caught:
                with embedding_build_transaction(budget, using="default"):
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT pg_sleep(0.15)")
            assert caught.value.__cause__.pgcode == "57014"
            assert limits()["statement_timeout"] == 30
            query()
            assert 0 < limits()["statement_timeout"] <= 30
        assert limits() == before
        query()
