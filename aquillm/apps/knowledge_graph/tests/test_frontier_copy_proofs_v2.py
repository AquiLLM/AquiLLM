"""No missing, stale or partially authenticated competing set can admit edges."""

import json

import pytest

from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2
from apps.knowledge_graph.projection.frontier_copy_batch_v2 import copy_heads
from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.tests.frontier_copy_batch_support import (
    CopyFixture,
    rewrite_index,
)
from apps.knowledge_graph.tests.frontier_fixture_support import key
from apps.knowledge_graph.tests.test_frontier_adversarial_v2 import run


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "foreign", "checksum", "forged_null"]
)
def test_invalid_index_batch_charged_once_and_final_authority_remains(mutation):
    fixture = CopyFixture()

    def transform(q, p, rows):
        if q != QUERIES["index_batch_multi"]:
            return rows
        if mutation == "missing":
            return rows[1:]
        if mutation == "duplicate":
            return (*rows, rows[0])
        if mutation == "foreign":
            rows[0]["generation_key"] = "f" * 64
        elif mutation == "checksum":
            rows[0]["record"]["payload_checksum"] = "0" * 64
        else:
            value = json.loads(rows[0]["record"]["payload_json"])
            value["payload"]["head"] = None
            rows[0]["record"]["payload_json"] = json.dumps(value)
        return rows

    fixture.driver.transform = transform
    with pytest.raises(TopologyLoadError):
        fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
    assert fixture.ledger.queries == len(fixture.driver.calls)
    assert fixture.ledger.source_rows == fixture.driver.scanned
    assert fixture.ledger._reserved_rows == 0 and not fixture.ledger._uncertain
    fixture.driver.transform = None
    fixture.session.final_attestation()
    assert not fixture.ledger._authority_reservations


@pytest.mark.parametrize(
    "mutation",
    [
        "successor_missing",
        "order",
        "family",
        "links",
        "predicate",
        "conflicting_refs",
        "directory_bounds",
    ],
)
def test_authenticated_copy_chain_corruption_never_reaches_raw_validation(mutation):
    fixture = CopyFixture(initialize=False)
    generation = fixture.bundles[0].generation.generation_key
    stream = "copies/" + fixture.seeds[0].identity_key
    prior_refs = []

    def mutate(kind, value):
        if (
            kind == "directory"
            and value["name"] == stream
            and mutation == "directory_bounds"
        ):
            value["name"] = ""
        if kind != "entry" or value["stream"] != stream:
            return
        if mutation == "successor_missing" and value["next"]:
            value["next"] = "f" * 64
        elif mutation == "order":
            value["order"] = ["a"]
        elif mutation == "family":
            value["family"] = "scope"
        elif mutation == "links":
            value["links"]["extra"] = None
        elif mutation == "predicate":
            value["links"]["relations"]["unknown"] = None
        elif mutation == "conflicting_refs":
            if prior_refs:
                ref = prior_refs[0].copy()
                ref[2] = "0" * 64
                value["refs"].append(ref)
            prior_refs.append(value["refs"][0])

    rewrite_index(fixture, generation, mutate)
    if mutation == "directory_bounds":
        # Scope may cross the corrupt directory; either stage must reject.
        with pytest.raises(TopologyLoadError):
            fixture.session.initialize()
            fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
        return
    fixture.session.initialize()
    before = len(fixture.driver.calls)
    with pytest.raises(TopologyLoadError):
        fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
    assert all(
        q != QUERIES["raw_batch_multi"] for q, _ in fixture.driver.calls[before:]
    )
    assert fixture.ledger.source_rows == fixture.driver.scanned


@pytest.mark.parametrize("kind", ["cycle", "depth", "bounds"])
def test_each_cached_directory_path_keeps_its_own_cycle_depth_and_bounds(kind):
    fixture = CopyFixture()
    reads = fixture.session.reads
    generation, identity = fixture.pairs[0]
    keys = [key("proof-path:" + str(i)) for i in range(34)]
    for i, index_key in enumerate(keys):
        child = keys[i + 1] if i + 1 < len(keys) else None
        if kind == "cycle" and i == 1:
            child = keys[0]
        name = "a" + str(i).zfill(3)
        if kind == "bounds" and i == 1:
            name = "a000"
        reads._index_records[generation, index_key] = (
            "directory",
            {"name": name, "head": None, "left": None, "right": child},
        )
    reads.manifests[generation]["frontier_root_key"] = keys[0]
    before = len(fixture.driver.calls)
    with pytest.raises(TopologyLoadError):
        copy_heads(fixture.session, ((generation, identity),))
    assert len(fixture.driver.calls) == before


def test_no_heap_consumption_until_all_first_heads_validate(monkeypatch):
    fixture = CopyFixture(initialize=False)

    def truncate_heads(q, p, rows):
        if q == QUERIES["index_batch_multi"] and any(
            json.loads(r["record"]["payload_json"])["payload"].get("family")
            == "relation"
            for r in rows
        ):
            assert len(rows) >= 2
            return rows[:-1]
        return rows

    fixture.driver.transform = truncate_heads

    def forbidden(*args, **kwargs):
        pytest.fail("incomplete competing heads admitted an edge")

    monkeypatch.setattr(FrontierSessionV2, "relation", forbidden)
    with pytest.raises(TopologyLoadError):
        run(fixture, fixture.driver, fixture.ledger)
    assert fixture.ledger.edges == 0
    assert fixture.ledger.queries == len(fixture.driver.calls)
    assert fixture.ledger.source_rows == fixture.driver.scanned


def test_original_shared_ledger_keeps_both_groups_final_reservations():
    fixture = CopyFixture()
    other = FrontierSessionV2(
        fixture.driver,
        group=fixture.group,
        request=fixture.request,
        budget=fixture.ledger,
        deadline=fixture.deadline,
    )
    other.initialize()
    assert len(fixture.ledger._authority_reservations) == 2
    while fixture.ledger.reserve_query():
        pass
    with pytest.raises(FrontierBudgetExceeded):
        fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
    fixture.session.final_attestation()
    other.final_attestation()
    assert fixture.ledger.queries == 64
    assert not fixture.ledger._authority_reservations


def test_cached_copy_cycle_rejects_before_reusing_any_raw_authority():
    fixture = CopyFixture()
    fixture.session.prepare_copies(fixture.pairs, relation_heads=False)
    stream = "copies/" + fixture.seeds[0].identity_key
    for (_, index_key), (kind, value) in fixture.session.reads._index_records.items():
        if kind == "entry" and value["stream"] == stream and value["next"] is None:
            value["next"] = index_key
    before = len(fixture.driver.calls)
    with pytest.raises(TopologyLoadError):
        fixture.session.prepare_copies(fixture.pairs, relation_heads=False)
    assert len(fixture.driver.calls) == before


def test_late_physical_copy_read_charged_once_and_does_not_continue():
    fixture = CopyFixture()
    before = len(fixture.driver.calls)

    def late(q, p, rows):
        fixture.session.reads.deadline = fixture.ledger.clock() - 1
        return rows

    fixture.driver.transform = late
    with pytest.raises(TimeoutError):
        fixture.session.prepare_copies(fixture.pairs, relation_heads=True)
    assert len(fixture.driver.calls) == before + 1 == fixture.ledger.queries
    assert fixture.ledger.source_rows == fixture.driver.scanned
    assert fixture.ledger._reserved_rows == 0 and not fixture.ledger._uncertain
