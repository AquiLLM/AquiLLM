"""Cross-generation authority must fit without trimming any selected document."""

import pytest

from apps.knowledge_graph.projection.frontier_adapter_v2 import page_from_rows
from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
)


def test_31_generations_530_documents_and_final_proof_fit_physical_budget():
    fixture = GenerationFixture()
    fixture.session.initialize()
    fixture.session.final_attestation()
    pages = [page_from_rows(rows) for rows in fixture.session.scope_rows.values()]
    assert len(pages) == 31
    assert sum(len(p.documents) for p in pages) == 530
    assert sum(len(p.provenance) for p in pages) == 561
    assert fixture.budget.queries == len(fixture.driver.calls) <= 32
    assert fixture.budget.source_rows == fixture.driver.scanned <= 5000
    assert fixture.budget._reserved_rows == 0
    assert all(bound <= 320 for bound in fixture.driver.bounds)


@pytest.mark.parametrize(
    "kind", ["manifest_batch", "index_batch_multi", "raw_batch_multi"]
)
@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "extra", "cross_generation"]
)
def test_batch_rejects_corrupt_authority_without_publishing_scope(kind, mutation):
    fixture = GenerationFixture(2, 4)

    def corrupt(query, params, rows):
        if query != QUERIES.get(kind):
            return rows
        if mutation == "missing":
            return rows[:-1]
        if mutation == "duplicate":
            return rows + (rows[-1],)
        if mutation == "extra":
            rows[-1]["lookup_key"] = "f" * 64
        else:
            rows[-1]["record"]["generation_key"] = "f" * 64
        return rows

    fixture.driver.transform = corrupt
    with pytest.raises(TopologyLoadError):
        fixture.session.initialize()
    assert fixture.session.scope_rows == {}
    assert fixture.budget.source_rows == fixture.driver.scanned


def test_final_manifest_batch_is_fresh_and_changed_marker_rejects():
    fixture = GenerationFixture(2, 4)
    fixture.session.initialize()
    marker = next(iter(fixture.driver.markers.values()))
    marker["membership_epoch"] += 1
    with pytest.raises(TopologyLoadError):
        fixture.session.final_attestation()
    assert fixture.driver.calls[-1][0] == QUERIES["manifest_batch"]


def test_final_rows_and_queries_survive_exploration_exhaustion():
    fixture = GenerationFixture(2, 4)
    fixture.session.initialize()
    budget = fixture.budget
    budget.charge_page(
        scanned_rows=budget.max_source_rows - budget.source_rows - 4, wire_bytes=0
    )
    with pytest.raises(FrontierBudgetExceeded):
        budget.reserve_read(1)
    fixture.session.final_attestation()
    assert budget.source_rows == 4998


def test_multiple_groups_share_one_nonrefundable_ledger():
    first = GenerationFixture(2, 4)
    second = GenerationFixture(2, 4, budget=first.budget)
    first.session.initialize()
    second.session.initialize()
    first.session.final_attestation()
    second.session.final_attestation()
    assert first.budget.queries == len(first.driver.calls) + len(second.driver.calls)
    assert first.budget.source_rows == first.driver.scanned + second.driver.scanned


def test_corrupt_index_batch_cannot_admit_authenticated_prefix():
    from apps.knowledge_graph.projection.frontier_generation_batch_v2 import indices_all

    fixture = GenerationFixture(2, 4)
    pairs = tuple(
        sorted(
            (row["generation_key"], key) for key, row in fixture.driver.index.items()
        )
    )

    def corrupt(query, params, rows):
        rows[-1]["record"]["index_key"] = "f" * 64
        return rows

    fixture.driver.transform = corrupt
    with pytest.raises(TopologyLoadError):
        indices_all(fixture.session.reads, pairs)
    assert fixture.session.reads._index_records == {}


def test_identical_raw_lookup_keys_in_different_generations_are_not_deduplicated():
    from dataclasses import asdict, replace

    from apps.knowledge_graph.projection.frontier_generation_batch_v2 import hydrate_all
    from apps.knowledge_graph.projection.frontier_index_contracts_v2 import (
        record_digest,
    )

    fixture = GenerationFixture(2, 4)
    key = "a" * 64
    expected = {}
    for selected in fixture.ready.selected_generations:
        generation = selected.generation_key
        row = replace(
            fixture.bundles[0].documents[0], document_key=key, generation_key=generation
        )
        fixture.driver.raw[generation, key] = [
            {
                "lookup_key": key,
                "record": {**asdict(row), "opaque_key": key},
                "labels": ["ProjectedRecord", "ProjectedDocument"],
            }
        ]
        expected[generation, "ProjectedDocument", key] = record_digest(row)
    result = hydrate_all(fixture.session.reads, expected)
    assert len(fixture.driver.calls) == 1 and fixture.budget.source_rows == 2
    assert len(fixture.driver.calls[0][1]["lookups_csv"].split(",")) == 2
    assert all(result[g]["ProjectedDocument", key].generation_key == g for g in result)


def test_raw_authority_is_fresh_even_when_index_cache_is_complete():
    fixture = GenerationFixture(2, 4)
    fixture.session.initialize()
    before = fixture.budget.queries

    def corrupt(query, params, rows):
        if query == QUERIES["raw_batch_multi"]:
            row = next(r for r in rows if "ArtifactProvenance" in r["labels"])
            row["record"]["source_hash"] = "f" * 63 + "e"
        return rows

    fixture.driver.transform = corrupt
    with pytest.raises(TopologyLoadError):
        fixture.session.initialize()
    assert fixture.session.scope_rows == {}
    assert fixture.budget.queries == before + 2


def test_final_query_reservation_survives_other_group_admission():
    fixture = GenerationFixture(2, 4)
    fixture.session.initialize()
    while fixture.budget.reserve_query():
        pass
    assert fixture.budget.queries == 63
    with pytest.raises(FrontierBudgetExceeded):
        GenerationFixture(2, 4, budget=fixture.budget).session.initialize()
    fixture.session.final_attestation()
    assert fixture.budget.queries == 64
