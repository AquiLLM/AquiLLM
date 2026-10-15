"""Genuine closed owner composition spends its one original nonempty ledger."""

from contextlib import ExitStack

import pytest
from django.db import connections

from apps.collections.tests.native_closed_seed_policy_fixtures import (
    native_frame,
    prepare,
)
from apps.knowledge_graph.retrieval.graph_runtime_seeds_v2 import (
    _group_request_native,
    group_request,
)
from apps.knowledge_graph.retrieval.native_direct_seed_repository import (
    _prepare_direct_native,
)
from apps.knowledge_graph.retrieval.production_direct import prepare_direct_seeds
from apps.knowledge_graph.tests.test_native_ontology_proof_postgres import (
    extractor_runtime,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


@pytest.mark.parametrize("native", [False, True])
def test_actual_nonempty_closed_seed_prefix_preserves_original_ledger(
    monkeypatch, native
):
    def runtime(graph):
        if not native:
            del graph._direct_seeds
        extractor_runtime(graph)

    with native_frame(monkeypatch, metadata=native, prepare_runtime=runtime) as frame:
        active, owner, _, groups, ledger = frame
        graph = active[1]
        identity = id(ledger)
        returns = []

        def observe(execute, sql, params, many, context):
            before = ledger.source_rows
            reserved = ledger._reserved_rows
            result = execute(sql, params, many, context)
            if reserved:
                assert ledger.source_rows == before + context["cursor"].rowcount
                returns.append(context["cursor"].rowcount)
            return result

        with ExitStack() as stack:
            for alias in ("default", "projection_source"):
                stack.enter_context(connections[alias].execute_wrapper(observe))
            for group in groups:
                kwargs = dict(
                    query="Atlas",
                    group=group,
                    authorization=graph.authorization,
                    read_budget=ledger,
                )
                (_group_request_native if native else group_request)(
                    **kwargs, **({"owner": owner} if native else {})
                )
            prepare(frame)
            for group in groups:
                if native:
                    _prepare_direct_native(
                        graph,
                        owner=owner,
                        group=group,
                        query="Atlas",
                        deadline=ledger.deadline,
                        read_budget=ledger,
                    )
                else:
                    prepare_direct_seeds(
                        graph,
                        query="Atlas",
                        scope=group.scope,
                        deadline=ledger.deadline,
                        read_budget=ledger,
                    )
        assert ledger.queries == len(returns) and ledger.source_rows == sum(returns)
        assert ledger.queries > 10 and ledger.source_rows > 10
        assert id(ledger) == identity and ledger.turn_budget is owner.stage.budget
        assert (
            ledger.max_queries == 64
            and ledger.max_source_rows == 5000
            and ledger.max_wire_bytes == 1048576
        )
        assert not ledger._uncertain and ledger._reserved_rows == 0
        assert owner.mapping and all(any(v.values()) for v in owner.mapping.values())
        print(
            {
                "native": native,
                "queries": ledger.queries,
                "rows": ledger.source_rows,
                "returns": returns,
            }
        )
