"""Actual client, ASGI router, finite worker and independent owner proof replay."""

import pytest

from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.retrieval.topology.frontier_acquisition_v2 import (
    acquire_frontier,
)
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.tests.frontier_scope_window_support import MixedScopeFixture
from apps.knowledge_graph.tests.frontier_task2_preparation_reference_v2 import (
    _prepare_copies_reference,
    _scope_rows_all_reference,
)
from apps.knowledge_graph.tests.test_frontier_copy_feasibility_v2 import GatewayOpener


def acquire(fixture, count):
    seeds = tuple(
        ProjectedSeedV1(b.relations[0].source_entity_key, 1.0 / count)
        for b in sorted(fixture.bundles, key=lambda b: b.generation.generation_key)[
            :count
        ]
    )
    fixture.session.initialize()
    fixture.driver.phase = "acquisition"
    try:
        acquire_frontier(
            fixture.session, fixture.assembly, seeds, fixture.caps, fixture.budget
        )
    except FrontierBudgetExceeded:
        assert count == 16
    fixture.driver.phase = "final"
    fixture.session.final_attestation()


@pytest.mark.parametrize("count", [1, 16])
def test_windows_preserve_exact_physical_plan_and_useful_final_proof(
    count, monkeypatch
):
    from apps.knowledge_graph.projection import (
        frontier_adapter_v2,
        frontier_copy_batch_v2,
    )

    reference = MixedScopeFixture()
    with monkeypatch.context() as old:
        old.setattr(frontier_adapter_v2, "scope_rows_all", _scope_rows_all_reference)
        old.setattr(frontier_copy_batch_v2, "prepare_copies", _prepare_copies_reference)
        acquire(reference, count)
    fixture = MixedScopeFixture()
    transport = GatewayOpener(fixture, monkeypatch, windows=True)
    acquire(fixture, count)
    assert fixture.driver.calls == reference.driver.calls
    assert fixture.driver.transcript == reference.driver.transcript
    assert fixture.assembly.candidates == reference.assembly.candidates
    assert (
        fixture.budget.queries,
        fixture.budget.source_rows,
        fixture.budget.edges,
        len(fixture.assembly.candidates),
    ) == ((59, 2830, 2, 1) if count == 1 else (64, 4660, 3, 3))
    assert fixture.budget.queries == len(fixture.driver.calls)
    assert fixture.budget.source_rows == fixture.driver.scanned
    assert len(transport.frames) < fixture.budget.queries
    assert (
        fixture.budget.wire_bytes == sum(f["wire"] for f in transport.frames) <= 1048576
    )
    assert transport.frames[-1]["wire"] == 10952
    assert not fixture.budget._authority_reservations and not fixture.budget._uncertain
    assert fixture.budget._window is None
    print(
        {
            "seeds": count,
            "queries": fixture.budget.queries,
            "rows": fixture.budget.source_rows,
            "RPCs": len(transport.frames),
            "wire": fixture.budget.wire_bytes,
            "requests": [f["request_bytes"] for f in transport.frames],
            "responses": [f["wire"] for f in transport.frames],
            "checkpoints": [f["checkpoint_bytes"] for f in transport.frames],
            "physical_per_RPC": [f["physical_queries"] for f in transport.frames],
            "window_frames_expanded_wire": [
                f["window_frames"] for f in transport.frames if f["window_frames"]
            ],
        }
    )
