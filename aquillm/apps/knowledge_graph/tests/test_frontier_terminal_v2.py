"""The hop horizon stops relations, not in-bound zero-hop copy/mention reads."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
    FrontierIndexFixtureDriver,
)
from apps.knowledge_graph.tests.frontier_terminal_fixture import terminal_bundles
from apps.knowledge_graph.tests.test_frontier_adversarial_v2 import run


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("other", [False, True])
def test_terminal_identity_acquires_own_or_other_generation_mention(
    monkeypatch, depth, other
):
    bundles, terminal, chunk, owner = terminal_bundles(depth, other)
    fixture = FrontierFixture(bundles)
    fixture.caps = replace(fixture.caps, max_depth=depth)
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    copy_calls, relation_streams = [], []
    prepare, entries = FrontierSessionV2.prepare_copies, FrontierSessionV2.entries

    def record_copies(session, pairs, *, relation_heads):
        copy_calls.extend(pairs)
        if any(identity == terminal for _, identity in pairs):
            assert not relation_heads
        return prepare(session, pairs, relation_heads=relation_heads)

    def record_entries(session, generation, head, stream):
        relation_streams.append(stream)
        yield from entries(session, generation, head, stream)

    monkeypatch.setattr(FrontierSessionV2, "prepare_copies", record_copies)
    monkeypatch.setattr(FrontierSessionV2, "entries", record_entries)
    result, budget = run(fixture, driver)
    assert chunk in {r.chunk_key for r in result.evidence_candidates}
    assert (fixture.bundles[owner].generation.generation_key, terminal) in copy_calls
    assert result.coverage.status == "complete"
    # Even fetching the terminal identity's relation head would exceed the hop
    # horizon. Its existing reverse edge is a concrete forbidden next-hop read.
    terminal_entities = {
        m.entity_key
        for b in fixture.bundles
        for m in b.automatic_memberships
        if (m.automatic_membership_key or m.entity_key) == terminal
    }
    assert not any(
        s.startswith("relations/" + entity + "/")
        for s in relation_streams
        for entity in terminal_entities
    )
    assert max(r.discovery_hop for r in result.snapshot.audit_rows) <= depth
    assert budget.queries == len(driver.calls) <= 64
    print("FRONTIER_TERMINAL", depth, other, budget.queries, budget.source_rows)


def test_insufficient_terminal_budget_is_partial_and_never_resets(monkeypatch):
    bundles, terminal, chunk, _ = terminal_bundles(2, False)
    fixture = FrontierFixture(bundles)
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    budget = fixture.budget(max_queries=52)
    from apps.knowledge_graph.projection.frontier_cypher_v2 import MANIFEST_BATCH_LOOKUP

    mention = FrontierSessionV2.mention
    reservations, terminal_calls = [], []
    limits = {
        name: value for name, value in vars(budget).items() if name.startswith("max_")
    }
    deadline, turn = budget.deadline, budget.turn_budget
    terminal_entities = {
        m.entity_key
        for bundle in fixture.bundles
        for m in bundle.automatic_memberships
        if (m.automatic_membership_key or m.entity_key) == terminal
    }

    def spend_shared_budget_before_terminal(session, generation, head, entity, rows):
        assert session.budget is budget
        if entity in terminal_entities:
            terminal_calls.append((len(driver.calls), budget.queries))
            # Other admitted work consumes the original shared ledger. Reserve
            # only the remaining non-manifest work, preserving final authority reads.
            while budget.reserve_query():
                reservations.append(True)
        yield from mention(session, generation, head, entity, rows)

    monkeypatch.setattr(
        FrontierSessionV2, "mention", spend_shared_budget_before_terminal
    )
    result, returned_budget = run(fixture, driver, budget)
    assert returned_budget is budget and len(terminal_calls) == 1
    assert reservations
    assert terminal in result.snapshot.identity_keys
    assert chunk not in {r.chunk_key for r in result.evidence_candidates}
    assert result.coverage.status == "partial"
    assert result.coverage.reasons == ("frontier_budget",)
    assert budget.queries == len(driver.calls) + len(reservations) == 52
    final_reads = driver.calls[terminal_calls[0][0] :]
    assert len(final_reads) == 1
    assert all(query == MANIFEST_BATCH_LOOKUP for query, _ in final_reads)
    assert {
        pair.split(":")[0]
        for _, params in final_reads
        for pair in params["lookups_csv"].split(",")
    } == {bundle.generation.generation_key for bundle in fixture.bundles}
    assert {name: getattr(budget, name) for name in limits} == limits
    assert budget.deadline == deadline and budget.turn_budget is turn
    assert budget.source_rows <= 5000
    assert budget.deadline == fixture.deadline
