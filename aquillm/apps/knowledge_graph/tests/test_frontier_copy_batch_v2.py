"""Batched copy preparation preserves canonical traversal, freshness and laziness."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2
from apps.knowledge_graph.projection.frontier_copy_batch_v2 import copy_heads
from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.retrieval.topology import (
    frontier_acquisition_v2 as acquisition,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.tests.frontier_copy_batch_support import CopyFixture
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationDriver,
)
from apps.knowledge_graph.tests.frontier_scalar_reference_v2 import (
    acquire_frontier as scalar,
)
from apps.knowledge_graph.tests.test_frontier_adversarial_v2 import run


@pytest.mark.parametrize("reverse", [False, True])
def test_complete_small_scalar_parity_multiple_copies_ties_global_bridge_and_absence(
    monkeypatch, reverse
):
    fixture = CopyFixture(initialize=False)
    fixture.seeds = (
        replace(fixture.seeds[0], mass=0.5),
        replace(fixture.seeds[0], identity_key="f" * 64, mass=0.5),
    )
    events = []
    original = FrontierSessionV2.relation

    def observed(session, generation, entry, **kwargs):
        events.append((generation, tuple(entry["order"]), kwargs["depth"]))
        yield from original(session, generation, entry, **kwargs)

    monkeypatch.setattr(FrontierSessionV2, "relation", observed)
    driver = GenerationDriver(fixture.bundles)
    if reverse:
        driver.transform = lambda q, p, rows: tuple(reversed(rows))
    actual, budget = run(fixture, driver)
    order = events[:]
    events.clear()
    monkeypatch.setattr(acquisition, "acquire_frontier", scalar)
    expected, old = run(fixture, GenerationDriver(fixture.bundles))
    assert actual.snapshot == expected.snapshot
    assert actual.evidence_candidates == expected.evidence_candidates
    assert actual.coverage == expected.coverage
    assert actual.coverage.status == "complete"
    assert fixture.answer_identity in actual.snapshot.identity_keys
    assert "f" * 64 not in actual.snapshot.identity_keys
    assert events == order and len(order) > 2
    assert budget.queries < old.queries < 64


def test_preparation_reads_complete_copy_chains_but_only_first_relation_heads():
    fixture = CopyFixture()
    session, driver = fixture.session, fixture.driver
    before = len(driver.calls)
    pairs = tuple((g, fixture.shared_identity) for g in sorted(session.selected))
    prepared = session.prepare_copies(pairs, relation_heads=True)
    copies = [copy for values in prepared.values() for copy, _ in values]
    assert len(copies) == 2
    reads = session.reads
    first_keys = {
        (g, h)
        for (g, _), values in prepared.items()
        for copy, _ in values
        for h in copy["links"]["relations"].values()
        if h
    }
    assert first_keys <= set(reads._index_records)
    successors = 0
    for generation, key in first_keys:
        entry = reads._index_records[generation, key][1]
        if entry["next"]:
            successors += 1
            assert (generation, entry["next"]) not in reads._index_records
        assert (generation, entry["links"]["evidence"]) not in reads._index_records
    assert all(query != QUERIES["raw"] for query, _ in driver.calls[before:])
    assert successors > 0
    assert fixture.ledger.queries == len(driver.calls)
    assert fixture.ledger.source_rows == driver.scanned
    session.final_attestation()


def test_horizon_preparation_fetches_no_relation_entry():
    fixture = CopyFixture()
    prepared = fixture.session.prepare_copies(fixture.pairs, relation_heads=False)
    assert sum(map(len, prepared.values())) == 2
    assert all(
        (g, head) not in fixture.session.reads._index_records
        for (g, _), values in prepared.items()
        for copy, _ in values
        for head in copy["links"]["relations"].values()
        if head
    )


@pytest.mark.parametrize("family", ["ProjectedEntity", "AutomaticMembership"])
def test_same_operation_repeated_at_later_depth_uses_fresh_raw_and_rejects_mutation(
    family,
):
    fixture = CopyFixture()
    prepared = fixture.session.prepare_copies(fixture.pairs, relation_heads=False)
    previous = len(fixture.driver.calls)
    again = fixture.session.prepare_copies(fixture.pairs, relation_heads=False)
    assert again == prepared
    assert all(
        q == QUERIES["raw_batch_multi"] for q, _ in fixture.driver.calls[previous:]
    )
    assert len(fixture.driver.calls) > previous

    def mutate(q, p, rows):
        if q == QUERIES["raw_batch_multi"]:
            row = next(r for r in rows if family in r["labels"])
            row["record"]["entity_key"] = "0" * 64
        return rows

    fixture.driver.transform = mutate
    with pytest.raises(TopologyLoadError):
        fixture.session.prepare_copies(fixture.pairs, relation_heads=False)
    assert fixture.ledger.source_rows == fixture.driver.scanned


def test_global_absence_is_authenticated_and_cached_work_obeys_deadline():
    fixture = CopyFixture()
    pairs = tuple((g, "f" * 64) for g in sorted(fixture.session.selected))
    assert copy_heads(fixture.session, pairs) == dict.fromkeys(pairs)
    before = len(fixture.driver.calls)
    assert copy_heads(fixture.session, pairs) == dict.fromkeys(pairs)
    assert len(fixture.driver.calls) == before
    fixture.session.reads.deadline = fixture.ledger.clock() - 1
    with pytest.raises(TimeoutError):
        copy_heads(fixture.session, pairs)
    assert len(fixture.driver.calls) == before


def test_actual_next_depth_copy_revalidates_raw_seen_in_previous_relation(monkeypatch):
    fixture = CopyFixture(initialize=False)
    original, calls = FrontierSessionV2.prepare_copies, []
    generation = fixture.bundles[0].generation.generation_key
    target = fixture.bundles[0].relations[0].target_entity_key

    def prepare(session, pairs, **kwargs):
        calls.append(pairs)
        if len(calls) == 2:
            for row in fixture.driver.raw[generation, target]:
                if "AutomaticMembership" in row["labels"]:
                    row["record"]["entity_key"] = "0" * 64
        return original(session, pairs, **kwargs)

    monkeypatch.setattr(FrontierSessionV2, "prepare_copies", prepare)
    with pytest.raises(TopologyLoadError):
        run(fixture, fixture.driver, fixture.ledger)
    assert len(calls) == 2 and fixture.ledger.edges > 0
    assert fixture.ledger.source_rows == fixture.driver.scanned
