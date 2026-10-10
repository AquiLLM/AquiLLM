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
    copies, entries = FrontierSessionV2.copies, FrontierSessionV2.entries

    def record_copies(session, generation, identity):
        copy_calls.append((generation, identity))
        yield from copies(session, generation, identity)

    def record_entries(session, generation, head, stream):
        relation_streams.append(stream)
        yield from entries(session, generation, head, stream)

    monkeypatch.setattr(FrontierSessionV2, "copies", record_copies)
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


def test_insufficient_terminal_budget_is_partial_and_never_resets():
    bundles, terminal, chunk, _ = terminal_bundles(2, False)
    fixture = FrontierFixture(bundles)
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    budget = fixture.budget(max_queries=52)
    result, _ = run(fixture, driver, budget)
    assert terminal in result.snapshot.identity_keys
    assert chunk not in {r.chunk_key for r in result.evidence_candidates}
    assert result.coverage.status == "partial"
    assert result.coverage.reasons == ("frontier_budget",)
    assert budget.queries == len(driver.calls) <= 52
    assert budget.source_rows <= 5000
    assert budget.deadline == fixture.deadline
