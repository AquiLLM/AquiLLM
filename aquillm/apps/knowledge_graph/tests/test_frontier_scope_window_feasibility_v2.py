"""Gate transport grouping on useful work under the unchanged physical caps."""

import pytest

from apps.knowledge_graph.projection.frontier_adapter_v2 import page_from_rows
from apps.knowledge_graph.retrieval.topology.frontier_acquisition_v2 import (
    acquire_frontier,
)
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.tests.frontier_scope_window_support import MixedScopeFixture


def test_mixed_full_scope_reaches_useful_traversal_and_final_proof():
    fixture = MixedScopeFixture()
    fixture.session.initialize()
    fixture.driver.phase = "acquisition"
    try:
        acquire_frontier(
            fixture.session,
            fixture.assembly,
            (fixture.seed,),
            fixture.caps,
            fixture.budget,
        )
    finally:
        fixture.driver.phase = "final"
        fixture.session.final_attestation()
    assert fixture.assembly.candidates
    assert fixture.budget.queries == len(fixture.driver.calls) <= 64
    assert fixture.budget.source_rows == fixture.driver.scanned <= 5000


def test_competing_copy_exhaustion_preserves_complete_scope_and_final_authority():
    from apps.knowledge_graph.tests.frontier_scalar_reference_v2 import (
        acquire_frontier as scalar_acquire,
    )

    fixture = MixedScopeFixture()
    fixture.session.initialize()
    pages = [page_from_rows(rows) for rows in fixture.session.scope_rows.values()]
    assert len(pages) == 31
    assert sum(len(page.documents) for page in pages) == 530
    assert sum(len(page.provenance) for page in pages) == 561
    fixture.driver.phase = "acquisition"
    with pytest.raises(FrontierBudgetExceeded, match="physical work exhausted"):
        scalar_acquire(
            fixture.session,
            fixture.assembly,
            (fixture.seed,),
            fixture.caps,
            fixture.budget,
        )
    assert fixture.budget.queries == 63
    assert fixture.budget.edges == 0
    assert fixture.assembly.candidates == {}
    fixture.driver.phase = "final"
    fixture.session.final_attestation()
    assert fixture.budget.queries == len(fixture.driver.calls) == 64
    assert fixture.budget.source_rows == fixture.driver.scanned == 2489
    assert fixture.driver.transcript[-1]["kind"] == "manifest_batch"
    assert fixture.driver.transcript[-1]["rows"] == 31
    assert not fixture.budget._authority_reservations
    assert not fixture.budget._uncertain
    assert all(
        frame["rows"] <= frame["maximum"] <= 320 for frame in fixture.driver.transcript
    )


def test_mixed_fixture_contains_root_bound_useful_relation_evidence():
    """Fixture validity control only; this is not the real acquisition gate."""
    fixture = MixedScopeFixture()
    fixture.session.initialize()
    generation = fixture.seed_generation
    copy, _ = next(fixture.session.copies(generation, fixture.seed.identity_key))
    stream = "relations/" + copy["order"][0] + "/knows"
    entry = next(
        fixture.session.entries(generation, copy["links"]["relations"]["knows"], stream)
    )
    pages = tuple(fixture.session.relation(generation, entry, stream=stream, depth=0))
    assert len(pages) == 1
    assert len(pages[0].relation_evidence) == 1
    assert len(pages[0].relations) == 1
    assert pages[0].relation_evidence[0].document_key in {
        row.document_key for row in fixture.ready.authorized_documents
    }
    fixture.session.final_attestation()
    assert fixture.budget.queries == len(fixture.driver.calls) <= 64
    assert fixture.budget.source_rows == fixture.driver.scanned <= 5000
