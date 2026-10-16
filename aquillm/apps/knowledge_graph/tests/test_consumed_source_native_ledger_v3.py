"""Genuine native SQL prefix and separately labeled canonical full-scope fixture."""

from contextlib import ExitStack

import pytest
from django.db import connections

from apps.collections.tests.native_closed_seed_policy_fixtures import (
    native_frame,
    prepare,
)
from apps.knowledge_graph.retrieval.graph_runtime_seeds_v2 import _group_request_native
from apps.knowledge_graph.retrieval.native_direct_seed_repository import (
    _prepare_direct_native,
)
from apps.knowledge_graph.tests.frontier_scope_window_support import MixedScopeFixture
from apps.knowledge_graph.tests.test_consumed_source_physical_v3 import run_physical
from apps.knowledge_graph.tests.test_native_ontology_proof_postgres import (
    extractor_runtime,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


def test_real_sql_prefix_keeps_original_ledger_through_consumed_frontier(monkeypatch):
    fixture = MixedScopeFixture()
    with native_frame(
        monkeypatch, metadata=True, prepare_runtime=extractor_runtime
    ) as frame:
        active, owner, _, groups, budget = frame
        graph, returns, maxima = active[1], [], []

        def observe(execute, sql, params, many, context):
            reserved = budget._reserved_rows
            result = execute(sql, params, many, context)
            if reserved:
                returns.append(context["cursor"].rowcount)
                maxima.append(reserved)
            return result

        with ExitStack() as stack:
            for alias in ("default", "projection_source"):
                stack.enter_context(connections[alias].execute_wrapper(observe))
            for group in groups:
                _group_request_native(
                    owner=owner,
                    query="Atlas",
                    group=group,
                    authorization=graph.authorization,
                    read_budget=budget,
                )
            prepare(frame)
            for group in groups:
                _prepare_direct_native(
                    graph,
                    owner=owner,
                    group=group,
                    query="Atlas",
                    deadline=budget.deadline,
                    read_budget=budget,
                )
        assert not owner._native_metadata.failed
        assert owner.mapping and all(any(v.values()) for v in owner.mapping.values())
        assert (
            (budget.queries, budget.source_rows)
            == (len(returns), sum(returns))
            == (18, 19)
        )
        assert 130 in maxima and budget.turn_budget is owner.stage.budget
        assert not budget._reserved_rows and not budget._uncertain
        fixture.budget = budget
        run_physical(
            fixture, monkeypatch, prefix="genuine-native-SQL/separate-canonical"
        )
        assert (
            fixture.session.budget is budget and fixture.session.reads.budget is budget
        )
        print(
            {"native_returned_rows": returns, "native_SQL_reservation_maxima": maxima}
        )
