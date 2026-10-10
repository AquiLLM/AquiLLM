"""Bounded intentional partiality never covers malformed authority/source rows."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.frontier_adapter_v2 import (
    Neo4jCanonicalFrontierAdapterV2,
)
from apps.knowledge_graph.projection.frontier_cypher_v2 import (
    MANIFEST_LOOKUP,
    RAW_LOOKUP,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_loader_v2 import (
    FrontierUnavailableV2,
    canonical_frontier_loader_v2,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
from apps.knowledge_graph.tests.frontier_hub_fixture import hub_bundles
from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
    FrontierIndexFixtureDriver,
)


def run(fixture, driver, budget=None):
    loader = canonical_frontier_loader_v2(
        selection=fixture.selection,
        codec=fixture.codec,
        adapter=Neo4jCanonicalFrontierAdapterV2(driver),
    )
    budget = budget or fixture.budget()
    result = loader.load(
        group=fixture.group,
        seeds=fixture.seeds,
        request=fixture.request,
        caps=fixture.caps,
        budget=budget,
        deadline=fixture.deadline,
    )
    return result, budget


def test_201_neighbor_hub_returns_partial_and_honestly_loses_late_answer():
    bundles, late_answer = hub_bundles()
    fixture = FrontierFixture(bundles)
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    result, budget = run(fixture, driver)
    assert result.coverage.status == "partial"
    assert result.coverage.reasons == ("frontier_budget",)
    assert result.coverage.exhaustive is False
    assert late_answer not in result.snapshot.identity_keys
    assert 1 < len(result.snapshot.identity_keys) <= 200
    assert budget.queries == len(driver.calls) <= 64
    assert budget.source_rows <= 5000


def test_preferred_predicate_wins_globally_even_when_its_quality_is_last():
    bundles, late_answer = hub_bundles(preferred_last=True)
    fixture = FrontierFixture(bundles)
    fixture.request = replace(
        fixture.request, predicate_weights=(("knows", 0.2), ("supports", 1.0))
    )
    result, budget = run(fixture, FrontierIndexFixtureDriver(fixture.bundles))
    assert late_answer in result.snapshot.identity_keys
    assert result.coverage.reasons == ("frontier_budget",)
    assert budget.queries <= 64


def test_query_budget_before_provenance_returns_typed_unavailable():
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    budget = fixture.budget(max_queries=1)
    with pytest.raises(FrontierUnavailableV2) as caught:
        run(fixture, driver, budget)
    assert caught.value.coverage.reasons == ("frontier_budget",)
    assert caught.value.accounting[3] == budget.queries == len(driver.calls) == 1


def test_missing_index_is_unavailable_but_partial_attestation_is_corruption():
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    for marker in driver.markers.values():
        for name in tuple(marker):
            if name.startswith("frontier_") or name == "validated_frontier_checksum":
                del marker[name]
    with pytest.raises(FrontierUnavailableV2) as caught:
        run(fixture, driver)
    assert caught.value.coverage.reasons == ("not_ready",)
    next(iter(driver.markers.values()))["frontier_version"] = (
        "canonical-frontier-index-v2"
    )
    with pytest.raises(TopologyLoadError):
        run(fixture, driver)


@pytest.mark.parametrize("mutation", ["duplicate", "tenth", "digest", "label"])
def test_invalid_raw_rows_including_overflow_are_charged_and_fail_closed(mutation):
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    original = driver.execute_read

    def read(query, parameters, **kwargs):
        rows = original(query, parameters, **kwargs)
        if query == RAW_LOOKUP and rows:
            if mutation == "duplicate":
                return (*rows, rows[0])
            if mutation == "tenth":
                return tuple(rows[0] for _ in range(10))
            first = rows[0]
            if mutation == "digest":
                first = {**first, "record": {**first["record"], "opaque_key": "f" * 64}}
            else:
                first = {**first, "labels": ["ProjectedRecord", "UnknownFamily"]}
            return (first, *rows[1:])
        return rows

    driver.execute_read = read
    budget = fixture.budget()
    with pytest.raises(TopologyLoadError):
        run(fixture, driver, budget)
    assert budget.source_rows >= 3


def test_authority_swap_on_late_manifest_discards_every_candidate():
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    original, counts = driver.execute_read, {}

    def read(query, parameters, **kwargs):
        rows = original(query, parameters, **kwargs)
        if query == MANIFEST_LOOKUP:
            generation = parameters["generation_key"]
            counts[generation] = counts.get(generation, 0) + 1
            if counts[generation] > 1:
                return ({"record": {**rows[0]["record"], "graph_checksum": "f" * 64}},)
        return rows

    driver.execute_read = read
    with pytest.raises(TopologyLoadError):
        run(fixture, driver)


def test_empty_requested_identity_never_returns_unrequested_neighbors():
    fixture = FrontierFixture()
    fixture.seeds = (replace(fixture.seeds[0], identity_key="f" * 64),)
    result, budget = run(fixture, FrontierIndexFixtureDriver(fixture.bundles))
    assert result.snapshot.identity_keys == ()
    assert result.evidence_candidates == ()
    assert len(result.snapshot.artifact_provenance) == 4
    assert budget.identities == 0


def test_second_round_cannot_reset_any_original_budget():
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    budget = fixture.budget()
    run(fixture, driver, budget)
    previous_rows, deadline = budget.source_rows, budget.deadline
    with pytest.raises(FrontierUnavailableV2):
        run(fixture, driver, budget)
    assert budget.source_rows > previous_rows
    assert budget.queries == len(driver.calls) <= 64
    assert budget.deadline == deadline


def test_late_manifest_is_charged_and_never_continues():
    from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
        FrontierBudgetV2,
    )
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    now = [1.0]
    budget = FrontierBudgetV2(
        turn_budget=TurnBudget(TurnLimits(), clock=lambda: now[0]),
        deadline=10.0,
        clock=lambda: now[0],
        max_identities=200,
        max_edges=1000,
        max_source_rows=5000,
        max_queries=64,
        max_wire_bytes=1048576,
    )
    original = driver.execute_read

    def read(*args, **kwargs):
        rows = original(*args, **kwargs)
        now[0] = 11.0
        return rows

    driver.execute_read = read
    with pytest.raises(FrontierUnavailableV2) as caught:
        run(fixture, driver, budget)
    assert caught.value.coverage.reasons == ("deadline",)
    assert budget.source_rows == budget.queries == len(driver.calls) == 1


def test_original_turn_cancellation_fences_loader_without_dispatch():
    from asyncio import CancelledError

    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    budget = fixture.budget()
    budget.turn_budget.close("cancelled")
    with pytest.raises(CancelledError):
        run(fixture, driver, budget)
    assert budget.queries == len(driver.calls) == 0
