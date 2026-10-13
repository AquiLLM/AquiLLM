"""Closed associations retain physical direction, provenance and finite work."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.frontier_evidence_v2 import page_associations
from apps.knowledge_graph.retrieval.graph_evidence_associations_v2 import (
    association_prefix,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.tests.graph_association_fixture_v2 import (
    AssociationFixture,
    key,
)


def fixture():
    f = AssociationFixture(("one", "two"))
    f.add("one", relations=(("a", "supports", "b"),), mentions=("c",))
    f.add("two", relations=(("a", "contradicts", "b"),), mentions=("d",))
    return f


def test_ref_requires_exact_complete_canonical_associations():
    ref = fixture().topology().evidence_candidates[0]
    for rows in (
        (),
        ref.associations[:-1],
        ref.associations * 2,
        tuple(reversed(ref.associations)),
        ref.associations * 1501,
    ):
        with pytest.raises(ValueError):
            replace(ref, associations=rows)
    with pytest.raises(TypeError):
        replace(ref, associations=list(ref.associations))
    for change in (
        {"identity_keys": (key("forged"),)},
        {"relation_types": ()},
        {"evidence_provenance_keys": (key("forged"),)},
    ):
        with pytest.raises(ValueError):
            replace(ref, **change)
    association = next(a for a in ref.associations if a.kind == "relation")
    with pytest.raises(ValueError):
        replace(association, direction="reverse_directed")
    with pytest.raises(ValueError):
        replace(association, kind="mention")


def test_reverse_traversal_preserves_scientific_endpoints_and_dedupes():
    f = AssociationFixture(("one",))
    f.add("one", relations=(("a", "supports", "b"),))
    f.add("one", relations=(("a", "supports", "b"),), reverse=True)
    refs = f.topology().evidence_candidates
    assert len(refs) == len(refs[0].associations) == 1
    a = refs[0].associations[0]
    assert (a.source_identity_key, a.target_identity_key, a.direction) == (
        key("a"),
        key("b"),
        "directed",
    )


def test_reversed_scientific_claim_does_not_pair_opposition():
    f = AssociationFixture(("answer", "support", "reverse"))
    f.add("answer", mentions=("a",), text=f.query)
    f.add("support", relations=(("a", "supports", "b"),), text="Other")
    f.add("reverse", relations=(("b", "contradicts", "a"),), text="Other")
    assert f.select(1).candidates[0].chunk_key == key("answer")


def test_undirected_physical_endpoints_are_canonical():
    f = AssociationFixture(("one",))
    f.add("one", relations=(("a", "knows", "b"),), reverse=True, direction="undirected")
    a = f.topology().evidence_candidates[0].associations[0]
    assert (a.source_identity_key, a.target_identity_key) == tuple(
        sorted((key("a"), key("b")))
    )
    assert a.direction == "undirected"


def test_producer_rejects_misaligned_physical_evidence_and_mention_origin():
    f = fixture()
    f.topology()
    page, entry, origin = f.pages[0]
    forged = replace(
        page,
        relation_evidence=(
            replace(page.relation_evidence[0], relation_type="contradicts"),
        ),
    )
    with pytest.raises(TopologyLoadError):
        page_associations(
            forged, generation=f.generation, ready=f.ready, origin=origin, entry=entry
        )
    mention, _, origin = f.pages[1]
    with pytest.raises(ValueError, match="admitted identity"):
        page_associations(
            mention, generation=f.generation, ready=f.ready, origin=key("other")
        )


def test_repeated_provenance_cannot_rebind_a_physical_chunk():
    f = fixture()
    f.topology()
    page, entry, origin = f.pages[0]
    rows, associations = page_associations(
        page, generation=f.generation, ready=f.ready, origin=origin, entry=entry
    )
    before = f.assembly.evidence_refs.build()
    with pytest.raises(ValueError, match="conflicting provenance"):
        f.assembly.evidence_refs._retain(
            (replace(rows[0], chunk_key=key("new-chunk")),), f.generation, associations
        )
    assert f.assembly.evidence_refs.build() == before


def test_association_prefix_keeps_whole_refs_across_groups():
    refs = fixture().topology().evidence_candidates
    refs = (refs[0], replace(refs[1], authority_group_checksum=key("other-group")))
    assert association_prefix(refs, maximum=3) == refs[:1]
    assert len(association_prefix(refs, maximum=4)) == 2


def test_batch_overflow_is_atomic_before_any_candidate_mutation():
    f = fixture()
    f.topology()
    accumulator = f.assembly.evidence_refs
    before = accumulator.build()
    accumulator.maximum = len(accumulator.provenance)
    page, entry, origin = f.pages[0]
    rows, associations = page_associations(
        page, generation=f.generation, ready=f.ready, origin=origin, entry=entry
    )
    with pytest.raises(FrontierBudgetExceeded):
        new_page = replace(
            page,
            relation_evidence=(replace(rows[0], evidence_key=key("new-row"),
                                      provenance_key=key("new-provenance")),),
        )
        accumulator.add_page(
            new_page, generation=f.generation, ready=f.ready, origin=origin, entry=entry
        )
    assert accumulator.build() == before


def test_real_loader_reports_association_overflow_as_partial(monkeypatch):
    from apps.knowledge_graph.projection.frontier_adapter_v2 import (
        Neo4jCanonicalFrontierAdapterV2,
    )
    from apps.knowledge_graph.projection.frontier_evidence_v2 import FrontierEvidenceV2
    from apps.knowledge_graph.retrieval.topology.frontier_loader_v2 import (
        canonical_frontier_loader_v2,
    )
    from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
    from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
        FrontierIndexFixtureDriver,
    )

    actual = FrontierEvidenceV2.__init__

    def stricter(self, binding, group, maximum):
        actual(self, binding, group, min(1, maximum))

    monkeypatch.setattr(FrontierEvidenceV2, "__init__", stricter)
    f = FrontierFixture()
    budget = f.budget()
    loader = canonical_frontier_loader_v2(
        selection=f.selection,
        codec=f.codec,
        adapter=Neo4jCanonicalFrontierAdapterV2(FrontierIndexFixtureDriver(f.bundles)),
    )
    result = loader.load(
        group=f.group,
        seeds=f.seeds,
        request=f.request,
        caps=f.caps,
        budget=budget,
        deadline=f.deadline,
    )
    assert sum(len(r.associations) for r in result.evidence_candidates) == 1
    assert (
        result.coverage.status == "partial"
        and "frontier_budget" in result.coverage.reasons
    )
    assert budget.source_rows > 1 and budget.queries > 1


def test_semantic_association_keys_preserve_physical_snapshot_and_ranking():
    f = AssociationFixture(("one",))
    f.add("one", relations=(("a", "supports", "b"),))
    row = f.pages[0][0].relation_evidence[0]
    topology = f.topology()
    ref, = topology.evidence_candidates
    assert row.provenance_key != row.evidence_key
    assert ref.evidence_provenance_keys == (row.provenance_key,)
    assert ref.associations[0].provenance_key == row.provenance_key
    physical, = topology.snapshot.relation_groups[0].evidence
    assert physical.provenance_key == row.evidence_key
    selected, = f.select().candidates
    assert selected.chunk_key == row.chunk_key
    assert selected.evidence_provenance_keys == (row.provenance_key,)
    assert selected.relation_types == ("supports",)
