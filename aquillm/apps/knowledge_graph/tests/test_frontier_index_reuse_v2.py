"""Component reuse accounting; real persisted G3 cases remain separate oracles."""
from collections import Counter
from copy import deepcopy

import pytest

from apps.knowledge_graph.projection.frontier_cypher_v2 import INDEX_LOOKUP, MANIFEST_LOOKUP, RAW_LOOKUP
from apps.knowledge_graph.projection.frontier_reads_v2 import FrontierReadsV2
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
from apps.knowledge_graph.tests.frontier_index_fixture_driver import FrontierIndexFixtureDriver
from apps.knowledge_graph.tests.test_frontier_adversarial_v2 import run


def reader(now=None):
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    budget = fixture.budget()
    deadline = fixture.deadline
    if now is not None:
        from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import FrontierBudgetV2
        from lib.retrieval.turn_budget import TurnBudget, TurnLimits
        budget = FrontierBudgetV2(turn_budget=TurnBudget(TurnLimits(), clock=lambda: now[0]),
            deadline=10.0, clock=lambda: now[0], max_identities=200, max_edges=1000,
            max_source_rows=5000, max_queries=64, max_wire_bytes=1048576)
        deadline = 9.0
    reads = FrontierReadsV2(driver, ready=fixture.group.scope.ready,
        budget=budget, deadline=deadline)
    selected = fixture.group.scope.ready.selected_generations[0]
    assert reads.attest(selected)
    return fixture, driver, budget, reads, selected


def root_key(driver, selected):
    return driver.markers[selected.generation_key]["frontier_root_key"]


def test_repeated_authenticated_index_key_performs_one_physical_read():
    _, driver, budget, reads, selected = reader()
    key = root_key(driver, selected)
    first = reads.index(selected.generation_key, key)
    original = deepcopy(first)
    first[1]["name"] = "caller cannot replace retained bytes"
    second = reads.index(selected.generation_key, key)
    assert second == original and second is not first
    calls = [p for q, p in driver.calls if q == INDEX_LOOKUP]
    assert calls == [{"index_key": key}]
    assert budget.queries == len(driver.calls) == 2


def test_whole_bridge_reuses_index_ancestors_but_attests_each_generation_twice():
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    result, budget = run(fixture, driver)
    assert fixture.answer_identity in result.snapshot.identity_keys
    keys = [p["index_key"] for q, p in driver.calls if q == INDEX_LOOKUP]
    assert keys and max(Counter(keys).values()) == 1
    manifests = Counter(p["generation_key"] for q, p in driver.calls if q == MANIFEST_LOOKUP)
    assert manifests == Counter({b.generation.generation_key: 2 for b in fixture.bundles})
    assert budget.queries == len(driver.calls) <= 64
    assert any(q == RAW_LOOKUP for q, _ in driver.calls)


@pytest.mark.parametrize("expired", ["deadline", "turn", "uncertain"])
def test_reuse_cannot_outlive_original_deadline_turn_or_accounting(expired):
    now = [1.0]
    _, driver, budget, reads, selected = reader(now)
    key = root_key(driver, selected)
    reads.index(selected.generation_key, key)
    before = len(driver.calls)
    if expired == "deadline":
        now[0] = 9.5
        error = TimeoutError
    elif expired == "turn":
        from asyncio import CancelledError
        budget.turn_budget.close("cancelled")
        error = CancelledError
    else:
        budget.stop_uncertain()
        error = TimeoutError
    with pytest.raises(error):
        reads.index(selected.generation_key, key)
    assert len(driver.calls) == before


@pytest.mark.parametrize("corruption", ["digest", "duplicate", "missing", "generation"])
def test_only_complete_authenticated_index_reads_can_be_reused(corruption):
    _, driver, budget, reads, selected = reader()
    key = root_key(driver, selected)
    original = driver.execute_read

    def invalid(query, parameters, **kwargs):
        rows = original(query, parameters, **kwargs)
        if query != INDEX_LOOKUP:
            return rows
        if corruption == "duplicate":
            return (*rows, rows[0])
        if corruption == "missing":
            return ()
        row = deepcopy(rows[0])
        field = "payload_checksum" if corruption == "digest" else "generation_key"
        row["record"][field] = "f" * 64
        return (row,)

    driver.execute_read = invalid
    with pytest.raises(TopologyLoadError):
        reads.index(selected.generation_key, key)
    driver.execute_read = original
    assert reads.index(selected.generation_key, key)[0] == "directory"
    assert budget.queries == len(driver.calls) == 3


def test_reuse_is_generation_scoped_and_never_reuses_final_authority():
    _, driver, _, reads, selected = reader()
    key = root_key(driver, selected)
    reads.index(selected.generation_key, key)
    with pytest.raises(TopologyLoadError):
        reads.index("f" * 64, key)
    driver.markers[selected.generation_key]["graph_checksum"] = "e" * 64
    with pytest.raises(TopologyLoadError):
        reads.attest(selected, final=True)
    assert sum(q == MANIFEST_LOOKUP for q, _ in driver.calls) == 2


def test_new_reader_does_not_reuse_another_invocations_index():
    fixture, driver, budget, reads, selected = reader()
    key = root_key(driver, selected)
    reads.index(selected.generation_key, key)
    other = FrontierReadsV2(driver, ready=fixture.group.scope.ready,
        budget=budget, deadline=fixture.deadline)
    assert other.attest(selected)
    other.index(selected.generation_key, key)
    assert sum(q == INDEX_LOOKUP for q, _ in driver.calls) == 2
    assert budget.queries == len(driver.calls) == 4


@pytest.mark.parametrize("corrupt", [False, True])
def test_cached_index_never_caches_raw_evidence_records(corrupt):
    _, driver, budget, reads, selected = reader()
    generation = selected.generation_key
    stream = "scope/" + selected.collection_key
    head = reads.head(generation, stream)
    entry = reads.entry(generation, head, stream=stream)
    expected = reads.hydrate(generation, entry["refs"])
    if corrupt:
        label, key, _ = entry["refs"][0]
        record = next(row for row in driver.raw[generation, key] if label in row["labels"])
        record["record"]["opaque_key"] = "f" * 64
        with pytest.raises(TopologyLoadError):
            reads.hydrate(generation, entry["refs"])
    else:
        assert reads.hydrate(generation, entry["refs"]) == expected
    assert sum(q == RAW_LOOKUP for q, _ in driver.calls) == 2
    assert budget.queries == len(driver.calls)


def test_unique_index_work_still_preserves_final_attestation_reservation():
    from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import FrontierBudgetExceeded
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    budget = fixture.budget(max_queries=4)
    selected = fixture.group.scope.ready.selected_generations[0]
    reads = FrontierReadsV2(driver, ready=fixture.group.scope.ready,
        budget=budget, deadline=fixture.deadline)
    assert reads.attest(selected)
    key = root_key(driver, selected)
    reads.index(selected.generation_key, key)
    other = next(key for key, row in driver.index.items()
        if row["generation_key"] == selected.generation_key and key != root_key(driver, selected))
    with pytest.raises(FrontierBudgetExceeded):
        reads.index(selected.generation_key, other)
    assert budget.queries == len(driver.calls) == 2
    assert reads.attest(selected, final=True)
    assert budget.queries == len(driver.calls) == 3
