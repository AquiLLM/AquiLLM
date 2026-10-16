"""Scope replay barriers, original final holds, and predispatch no-work cases."""

from copy import deepcopy
from time import monotonic

import pytest

from apps.knowledge_graph.projection.frontier_preparation_reads_v2 import OwnerBoundary
from apps.knowledge_graph.projection.frontier_reads_v2 import FrontierReadsV2
from apps.knowledge_graph.projection.frontier_scope_complete_v3 import (
    ScopeCompletePreparation,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_checkpoint_v2 import (
    restore,
    snapshot,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_owner_v2 import (
    run_preparation,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_worker_v2 import (
    PlanningCursor,
)
from apps.knowledge_graph.tests.frontier_scope_complete_support import scope_state
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener


def test_partial_scope_wave_retains_cached_guard_and_cannot_publish_corrupt_tail():
    fixture, state = scope_state(generations=2, documents=200)
    cache = deepcopy(state.reads._index_records)
    effect = state.next_read(2)
    assert len(effect.pairs) == 1
    state.apply_result(state.reads.read_batch(effect.kind, effect.pairs))
    assert state.reads._index_records == cache
    operation, checkpoint = snapshot(state)
    cursor = PlanningCursor(5000, monotonic() + 5)
    reads = FrontierReadsV2(
        None, ready=fixture.ready, budget=cursor, deadline=cursor.deadline
    )
    worker = restore(reads, operation, checkpoint, complete_scope=True)
    with pytest.raises(ValueError):
        restore(reads, operation, checkpoint)
    assert worker.guard == set(state.staged)
    effect = worker.next_read()
    physical = state.reads.read_batch(effect.kind, effect.pairs)
    worker.apply_result(physical)
    with pytest.raises(OwnerBoundary):
        worker.next_read()
    assert not reads._index_records
    state.next_read()
    with pytest.raises(TopologyLoadError):
        state.apply_result(())
    assert state.reads._index_records == cache


def test_no_queries_left_never_dispatches_and_retains_final_hold(monkeypatch):
    fixture, state = scope_state(stage="scope_raw")
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    while fixture.budget.reserve_query():
        pass
    with pytest.raises(FrontierBudgetExceeded):
        run_preparation(state)
    assert not transport.requests and not fixture.budget._uncertain
    fixture.session.final_attestation()
    assert fixture.budget.queries == 64


def test_second_group_authority_survives_scope_window_and_first_final(monkeypatch):
    fixture, state = scope_state(stage="scope_raw")
    WindowOpener(fixture, monkeypatch, complete_scope=True)
    other = fixture.budget.reserve_authority(4, 1)
    fixture.budget.reserve_authority_wire(other, 17)
    run_preparation(state)
    assert fixture.budget._authority_reservations[other] == (4, 1, 17)
    fixture.session.final_attestation()
    assert fixture.budget._authority_reservations == {other: (4, 1, 17)}


def test_no_read_scope_avoids_empty_window_and_final_revocation_still_fails(
    monkeypatch,
):
    fixture, state = scope_state(stage="scope_raw")
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    assert run_preparation(ScopeCompletePreparation(state.reads, {})) == {}
    assert not transport.requests
    assert run_preparation(state)
    marker = next(iter(fixture.driver.markers.values()))
    marker["frontier_root_key"] = "0" * 64
    with pytest.raises(TopologyLoadError):
        fixture.session.final_attestation()
    assert fixture.budget.queries == len(fixture.driver.calls)
    assert fixture.budget.source_rows == fixture.driver.scanned


@pytest.mark.parametrize(
    "disabled", ["binary_batches", "preparation_windows", "complete_scope_windows"]
)
def test_scope_capability_selects_legacy_before_initial_authority(
    monkeypatch, disabled
):
    from apps.knowledge_graph.retrieval.topology.frontier_gateway_client_v2 import (
        FrontierGatewayClientV2,
    )
    from apps.knowledge_graph.tests.frontier_generation_batch_support import (
        GenerationFixture,
    )

    fixture = GenerationFixture(2, 5)
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    fixture.session.reads.driver = FrontierGatewayClientV2(
        fixture.session.reads.driver.client, **{disabled: False}
    )
    fixture.session.initialize()
    assert fixture.session.scope_rows
    assert all("/v3/" not in request.full_url for request in transport.requests)
    assert not fixture.session.reads.complete_scope_windows
    fixture.session.final_attestation()


def test_single_generation_canonical_client_selects_complete_scope_from_directory(
    monkeypatch,
):
    from apps.knowledge_graph.tests.frontier_generation_batch_support import (
        GenerationFixture,
    )

    fixture = GenerationFixture(1, 10)
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    fixture.session.initialize()
    assert sum(map(len, fixture.session.scope_rows.values())) == 21
    assert any(
        "/v3/topology/frontier/scope-window" in request.full_url
        for request in transport.requests
    )
    fixture.session.final_attestation()
    assert not fixture.budget._authority_reservations and not fixture.budget._uncertain
