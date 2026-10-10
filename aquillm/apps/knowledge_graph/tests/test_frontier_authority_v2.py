"""Unauthorized observations cannot become zero-hop or relation transit."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.frontier_cypher_v2 import INDEX_LOOKUP
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.tests.frontier_authority_fixture import (
    inaccessible_bridge_bundles,
)
from apps.knowledge_graph.tests.frontier_fixture_support import (
    FrontierFixture,
    bridge_bundles,
)
from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
    FrontierIndexFixtureDriver,
)
from apps.knowledge_graph.tests.test_frontier_adversarial_v2 import run


def test_inaccessible_relation_cannot_reseed_canonical_copy_in_other_generation():
    fixture = FrontierFixture(inaccessible_bridge_bundles())
    result, budget = run(fixture, FrontierIndexFixtureDriver(fixture.bundles))
    assert result.snapshot.identity_keys == (fixture.seeds[0].identity_key,)
    assert fixture.shared_identity not in result.snapshot.identity_keys
    assert fixture.answer_identity not in result.snapshot.identity_keys
    assert result.snapshot.relation_groups == ()
    assert budget.edges == 0


@pytest.mark.parametrize("mutation", ["duplicate", "hash", "missing"])
def test_index_duplicate_sentinel_dangling_or_changed_link_fails_closed(mutation):
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    original = driver.execute_read

    def read(query, parameters, **kwargs):
        rows = original(query, parameters, **kwargs)
        if query == INDEX_LOOKUP and rows:
            if mutation == "duplicate":
                return (*rows, rows[0])
            if mutation == "missing":
                return ()
            return ({"record": {**rows[0]["record"], "payload_json": "{}"}},)
        return rows

    driver.execute_read = read
    budget = fixture.budget()
    with pytest.raises(TopologyLoadError):
        run(fixture, driver, budget)
    assert budget.queries == len(driver.calls)
    assert budget.source_rows >= 2


def test_no_cross_request_cache_can_hide_late_generation_revocation():
    fixture = FrontierFixture()
    driver = FrontierIndexFixtureDriver(fixture.bundles)
    run(fixture, driver)
    next(iter(driver.markers.values()))["state"] = "failed"
    with pytest.raises(TopologyLoadError):
        run(fixture, driver)


def test_zero_weight_relation_cannot_admit_a_target_or_invalid_snapshot_group():
    first, second = bridge_bundles()
    first = replace(
        first, evidence=tuple(replace(r, confidence=0.0) for r in first.evidence)
    )
    fixture = FrontierFixture((first, second))
    result, budget = run(fixture, FrontierIndexFixtureDriver(fixture.bundles))
    assert result.snapshot.identity_keys == (fixture.seeds[0].identity_key,)
    assert result.snapshot.relation_groups == ()
    assert budget.edges == 0
