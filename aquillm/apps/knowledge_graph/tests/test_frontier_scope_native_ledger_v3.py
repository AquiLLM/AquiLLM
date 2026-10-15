"""Actual Task12 SQL and a separately labeled canonical frontier share one ledger.

The native corpus has two groups; canonical31/530 is an independent fixture.
Root's ordinary native254 replay closes representation/body/provenance integration.
"""

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
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.retrieval.topology.frontier_acquisition_v2 import (
    acquire_frontier,
)
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.tests.frontier_scope_window_support import MixedScopeFixture
from apps.knowledge_graph.tests.test_frontier_copy_feasibility_v2 import GatewayOpener
from apps.knowledge_graph.tests.test_native_ontology_proof_postgres import (
    extractor_runtime,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


def test_real_native_sql_prefix_flows_into_canonical_on_original_ledger(monkeypatch):
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
        inherited = budget.queries, budget.source_rows
        assert inherited == (len(returns), sum(returns)) == (18, 19)
        assert 130 in maxima
        assert budget.turn_budget is owner.stage.budget
        assert not budget._reserved_rows and not budget._uncertain
        # Explicit separated-fixture boundary: exact original object, clocks and caps.
        fixture.budget = fixture.session.budget = fixture.session.reads.budget = budget
        fixture.session.reads.deadline = budget.deadline
        transport = GatewayOpener(
            fixture, monkeypatch, windows=True, complete_scope=True
        )
        seeds = tuple(
            ProjectedSeedV1(b.relations[0].source_entity_key, 1 / 16)
            for b in sorted(fixture.bundles, key=lambda b: b.generation.generation_key)[
                :16
            ]
        )
        fixture.session.initialize()
        fixture.driver.phase = "acquisition"
        try:
            acquire_frontier(
                fixture.session, fixture.assembly, seeds, fixture.caps, budget
            )
        except FrontierBudgetExceeded:
            pass
        fixture.driver.phase = "final"
        fixture.session.final_attestation()
        assert fixture.session.reads.budget is budget
        assert {s.identity_key for s in seeds} <= budget._identities
        assert fixture.assembly.candidates
        assert budget.queries == inherited[0] + len(fixture.driver.calls) <= 64
        assert budget.source_rows == inherited[1] + fixture.driver.scanned <= 5000
        assert budget.wire_bytes == sum(f["wire"] for f in transport.frames) <= 1048576
        assert not budget._uncertain and not budget._authority_reservations
        assert budget._window is None and not budget._reserved_rows
        print(
            {
                "native_prefix": inherited,
                "returns": returns,
                "SQL_reservations": maxima,
                "queries": budget.queries,
                "rows": budget.source_rows,
                "wire": budget.wire_bytes,
                "identities": budget.identities,
                "candidates": len(fixture.assembly.candidates),
            }
        )
