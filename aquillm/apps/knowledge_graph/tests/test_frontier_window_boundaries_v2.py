"""Owner barriers preserve cache-first waves, fresh raw dependencies and fallback."""

from copy import deepcopy
from time import monotonic

import pytest

from apps.knowledge_graph.projection.frontier_copy_preparation_v2 import CopyPreparation
from apps.knowledge_graph.projection.frontier_preparation_reads_v2 import OwnerBoundary
from apps.knowledge_graph.projection.frontier_reads_v2 import FrontierReadsV2
from apps.knowledge_graph.retrieval.topology import (
    frontier_window_contracts_v2 as contracts,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_checkpoint_v2 import (
    restore,
    snapshot,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_worker_v2 import (
    PlanningCursor,
)
from apps.knowledge_graph.tests.frontier_copy_batch_support import CopyFixture
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener


def test_nonprefix_cached_children_stop_worker_without_rereads(monkeypatch):
    expected, actual = CopyFixture(), CopyFixture()
    for fixture in (expected, actual):
        fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
        for g, marker in fixture.session.reads.manifests.items():
            fixture.session.reads._index_records.pop((g, marker["frontier_root_key"]))
    result = expected.session.prepare_copies(expected.pairs, relation_heads=True)
    transport = WindowOpener(actual, monkeypatch)
    assert actual.session.prepare_copies(actual.pairs, relation_heads=True) == result
    assert actual.driver.calls == expected.driver.calls
    assert len(transport.requests) == 2
    assert actual.ledger.queries == len(actual.driver.calls)


def test_all_cached_authenticated_absence_has_no_window_or_physical_read(monkeypatch):
    fixture = CopyFixture()
    pairs = tuple((g, "f" * 64) for g in fixture.session.selected)
    assert fixture.session.prepare_copies(pairs, relation_heads=True) == dict.fromkeys(
        pairs, []
    )
    before = len(fixture.driver.calls)
    transport = WindowOpener(fixture, monkeypatch)
    assert fixture.session.prepare_copies(pairs, relation_heads=True) == dict.fromkeys(
        pairs, []
    )
    assert not transport.requests and len(fixture.driver.calls) == before


def test_oversized_checkpoint_selects_unchanged_single_reads_before_any_grant(
    monkeypatch,
):
    expected, actual = CopyFixture(), CopyFixture()
    result = expected.session.prepare_copies(expected.pairs, relation_heads=True)
    transport = WindowOpener(actual, monkeypatch)
    monkeypatch.setattr(contracts, "MAX_REQUEST_BYTES", 1)
    assert actual.session.prepare_copies(actual.pairs, relation_heads=True) == result
    assert actual.driver.calls == expected.driver.calls
    assert all(r.full_url.endswith("/batch-binary") for r in transport.requests)
    assert actual.ledger._window is None and not actual.ledger._uncertain


def test_copy_raw_checkpoint_requires_owner_before_heads_when_prior_raw_is_local():
    fixture = CopyFixture()
    state = CopyPreparation(fixture.session, fixture.pairs, relation_heads=True)
    while (effect := state.next_read()).kind != "raw_batch_multi":
        state.apply_result(fixture.session.reads.read_batch(effect.kind, effect.pairs))
    # Exercise an earlier raw physical prefix retained by the owner. The fixed
    # planner supports this cut when the captured available-row allowance is ten.
    effect = state.next_read(10)
    state.apply_result(fixture.session.reads.read_batch(effect.kind, effect.pairs))
    assert state.raw_pending
    operation, checkpoint = snapshot(state)
    assert checkpoint["earlier_raw"] is True
    assert "fresh" not in checkpoint
    cursor = PlanningCursor(5000, monotonic() + 5)
    worker_reads = FrontierReadsV2(
        None, ready=fixture.session.ready, budget=cursor, deadline=cursor.deadline
    )
    worker = restore(worker_reads, operation, checkpoint)
    while True:
        try:
            effect = worker.next_read()
        except OwnerBoundary:
            break
        assert effect.kind == "raw_batch_multi"
        physical = fixture.session.reads.read_batch(effect.kind, effect.pairs)
        worker.apply_result(physical)
        state.next_read()
        state.apply_result(physical)
    assert worker.stage == "copy_raw" and worker.first == []
    assert state.next_read().kind == "index_batch_multi"


def test_partial_index_wave_never_publishes_on_later_corruption():
    fixture = CopyFixture()
    state = CopyPreparation(fixture.session, fixture.pairs, relation_heads=True)
    before = deepcopy(fixture.session.reads._index_records)
    effect = state.next_read(2)
    assert len(effect.pairs) == 1
    state.apply_result(fixture.session.reads.read_batch(effect.kind, effect.pairs))
    assert fixture.session.reads._index_records == before
    effect = state.next_read(2)
    assert effect.kind == "index_batch_multi"
    with pytest.raises(Exception):
        state.apply_result(())
    assert fixture.session.reads._index_records == before


def test_window_horizon_never_reads_first_heads(monkeypatch):
    fixture = CopyFixture()
    WindowOpener(fixture, monkeypatch)
    prepared = fixture.session.prepare_copies(fixture.pairs, relation_heads=False)
    assert sum(map(len, prepared.values())) == 2
    assert all(
        (g, head) not in fixture.session.reads._index_records
        for (g, _), copies in prepared.items()
        for entry, _ in copies
        for head in entry["links"]["relations"].values()
        if head
    )
