"""Original 32-document admission must leave room for real frontier work."""

import pytest

from apps.knowledge_graph.projection.frontier_adapter_v2 import page_from_rows
from apps.knowledge_graph.projection.frontier_cypher_v2 import MANIFEST_LOOKUP, RAW_LOOKUP
from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.tests.frontier_scope_batch_support import ScopeBatchFixture


def test_actual_32_document_scope_fits_original_ledger_with_fresh_final_manifest():
    fixture = ScopeBatchFixture()
    budget, turn, deadline = fixture.budget, fixture.budget.turn_budget, fixture.deadline
    fixture.session.initialize()
    generation = fixture.bundle.generation.generation_key
    page = page_from_rows(fixture.session.scope_rows[generation])
    assert {r.document_key for r in page.documents} == {
        r.document_key for r in fixture.group.scope.ready.authorized_documents
    }
    assert len(page.documents) == 32 and len(page.provenance) == 33
    fixture.session.final_attestation()
    assert fixture.session.budget is budget and budget.turn_budget is turn
    assert budget.deadline == deadline and budget.max_queries == 64
    assert (budget.max_source_rows, budget.max_wire_bytes) == (5000, 1048576)
    assert budget.queries == len(fixture.driver.calls) <= 32
    assert budget.source_rows == fixture.driver.scanned
    assert budget._reserved_rows == 0 and not budget._uncertain
    assert [q for q, _ in fixture.driver.calls].count(MANIFEST_LOOKUP) == 2
    assert fixture.driver.calls[-1][0] == MANIFEST_LOOKUP
    batches = [(q, params) for q, params in fixture.driver.calls if q == QUERIES.get("index_batch")]
    assert batches and all(1 <= len(params["keys_csv"].split(",")) <= 32 for _, params in batches)
    raw = [params for q, params in fixture.driver.calls if q == RAW_LOOKUP]
    assert len(raw) == 2
    assert all(1 <= len(params["keys_csv"].split(",")) <= 32 for params in raw)


def test_scope_reordering_does_not_change_authenticated_document_coverage():
    fixture = ScopeBatchFixture()
    fixture.driver.transform = lambda q, p, rows: tuple(reversed(rows))
    fixture.session.initialize()
    page = page_from_rows(fixture.session.scope_rows[fixture.bundle.generation.generation_key])
    assert len(page.documents) == 32 and len(page.provenance) == 33
    fixture.session.final_attestation()
    assert fixture.budget.source_rows == fixture.driver.scanned


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "source_hash", "foreign_generation"])
def test_scope_raw_records_remain_fresh_and_complete(mutation):
    fixture = ScopeBatchFixture(2)
    fixture.session.initialize()
    generation = fixture.bundle.generation.generation_key
    before = fixture.budget.queries

    def transform(query, params, rows):
        if query != RAW_LOOKUP:
            return rows
        if mutation == "missing":
            return rows[:-1]
        if mutation == "duplicate":
            return rows + (rows[-1],)
        row = next(r for r in rows if "ArtifactProvenance" in r["labels"])
        row["record"]["source_hash" if mutation == "source_hash" else "generation_key"] = "0" * 64
        return rows

    fixture.driver.transform = transform
    with pytest.raises(TopologyLoadError):
        fixture.session.initialize()
    assert fixture.budget.queries > before
    assert fixture.budget.source_rows == fixture.driver.scanned
    assert fixture.budget._reserved_rows == 0


def test_final_manifest_is_fresh_after_batched_scope_admission():
    fixture = ScopeBatchFixture()
    fixture.session.initialize()
    generation = fixture.bundle.generation.generation_key
    fixture.driver.markers[generation]["graph_checksum"] = "0" * 64
    before = fixture.budget.queries
    with pytest.raises(TopologyLoadError):
        fixture.session.final_attestation()
    assert fixture.budget.queries == before + 1
    assert fixture.driver.calls[-1][0] == MANIFEST_LOOKUP


def test_actual_loader_retains_evidence_after_complete_32_document_admission():
    from apps.knowledge_graph.projection.frontier_adapter_v2 import Neo4jCanonicalFrontierAdapterV2
    from apps.knowledge_graph.retrieval.topology.frontier_loader_v2 import canonical_frontier_loader_v2

    fixture = ScopeBatchFixture()
    loader = canonical_frontier_loader_v2(
        selection=fixture.selection, codec=fixture.codec,
        adapter=Neo4jCanonicalFrontierAdapterV2(fixture.driver),
    )
    result = loader.load(
        group=fixture.group, seeds=fixture.seeds, request=fixture.request,
        caps=fixture.caps, budget=fixture.budget, deadline=fixture.deadline,
    )
    assert result.coverage.status == "complete"
    assert result.coverage.ready_documents == result.coverage.selected_documents == 32
    assert fixture.bundle.chunks[0].chunk_key in {r.chunk_key for r in result.evidence_candidates}
    assert fixture.budget.queries == len(fixture.driver.calls) <= 64
    assert fixture.budget.source_rows == fixture.driver.scanned
    assert fixture.driver.calls[-1][0] == MANIFEST_LOOKUP
