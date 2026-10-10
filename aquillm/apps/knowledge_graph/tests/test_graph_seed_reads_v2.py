"""Actual PostgreSQL data reads reserve first and retain late/failed charges."""

import pytest
from django.db import DatabaseError, connection

from apps.collections.models import Collection
from apps.knowledge_graph.retrieval.graph_seed_reads_v2 import seed_query_rows
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture

pytestmark = pytest.mark.django_db


def source():
    rows = [Collection.objects.create(name=f"seed-{i}") for i in range(3)]
    return Collection.objects.filter(pk__in=[r.pk for r in rows]).values_list(
        "pk", flat=True
    )


def test_exact_known_data_query_charges_returned_rows_and_one_statement():
    query = source()
    budget = FrontierFixture().budget()
    rows = seed_query_rows(query, maximum=2, budget=budget)
    assert len(rows) == 2
    assert budget.queries == 1 and budget.source_rows == 2
    assert budget._reserved_rows == 0 and not budget._uncertain


def test_low_allowance_prevents_data_dispatch():
    query = source()
    budget = FrontierFixture().budget(max_source_rows=1)
    with pytest.raises(FrontierBudgetExceeded):
        seed_query_rows(query, maximum=2, budget=budget)
    assert budget.queries == budget.source_rows == 0


def test_late_and_invalid_work_keeps_actual_charges():
    from asyncio import CancelledError

    query = source()
    budget = FrontierFixture().budget()
    sql, _ = query[:2].query.sql_with_params()

    def late(execute, statement, params, many, context):
        result = execute(statement, params, many, context)
        if statement == sql:
            budget.turn_budget.close("test-late")
        return result

    with connection.execute_wrapper(late), pytest.raises(CancelledError):
        seed_query_rows(query, maximum=2, budget=budget)
    assert budget.queries == 1 and budget.source_rows == 2


def test_failed_unknown_read_poisoning_does_not_block_transaction_cleanup():
    query = source()
    compiled, _ = query[:2].query.sql_with_params()
    budget = FrontierFixture().budget()

    def fail(execute, statement, params, many, context):
        if statement == compiled:
            raise DatabaseError("synthetic seed read failure")
        return execute(statement, params, many, context)

    with connection.execute_wrapper(fail), pytest.raises(DatabaseError):
        seed_query_rows(query, maximum=2, budget=budget)
    assert budget.queries == 1 and budget._reserved_rows == 2
    assert budget._uncertain
    assert Collection.objects.count() == 3
