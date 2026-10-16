"""Used evidence remains fail closed under the exact demand-closure version."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.frontier_cypher_v2 import MANIFEST_BATCH_LOOKUP
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_loader_v2 import (
    FrontierUnavailableV2,
)
from apps.knowledge_graph.tests.consumed_source_test_support import (
    acquire_fixture,
    multi_document_fixture,
)
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
from apps.knowledge_graph.tests.test_consumed_source_closure_v3 import loaded


@pytest.mark.parametrize(
    "family,field",
    [
        ("ProjectedDocument", "generation_key"),
        ("ProjectedChunk", "document_key"),
        ("ProjectedChunk", "chunk_number"),
        ("ArtifactProvenance", "collection_key"),
        ("ArtifactProvenance", "scope_key"),
        ("ProjectedRelation", "artifact_key"),
        ("ProjectedEntity", "collection_key"),
        ("AutomaticMembership", "decision_checksum"),
        ("ProjectedEvidence", "document_key"),
        ("ProjectedEvidence", "artifact_key"),
    ],
)
def test_wrong_or_swapped_consumed_raw_proof_rejects(family, field):
    f = FrontierFixture()
    driver = FrontierIndexFixtureDriver(f.bundles)
    bundle = f.bundles[0]
    changed = []
    for (generation, _), rows in driver.raw.items():
        if generation == bundle.generation.generation_key:
            for row in rows:
                if family in row["labels"]:
                    row["record"][field] = 999 if field == "chunk_number" else "a" * 64
                    changed.append(True)
    assert changed
    budget = f.budget()
    with pytest.raises(TopologyLoadError):
        loaded(f, driver, budget=budget)
    assert budget.queries == len(driver.calls) and budget.source_rows


@pytest.mark.parametrize(
    "family", ["ProjectedDocument", "ProjectedChunk", "ArtifactProvenance"]
)
def test_mention_only_consumed_document_chunk_and_provenance_are_fresh(family):
    f = multi_document_fixture(mention_only=True)
    bundle = f.bundles[0]
    document = bundle.entity_mentions[0].document_key
    chunk = bundle.entity_mentions[0].chunk_key
    lookup = chunk if family == "ProjectedChunk" else document
    f.driver.raw[bundle.generation.generation_key, lookup] = [
        row
        for row in f.driver.raw[bundle.generation.generation_key, lookup]
        if family not in row["labels"]
    ]
    with pytest.raises(TopologyLoadError):
        acquire_fixture(f)
    assert document not in {row.document_key for _, row in f.assembly.mentions.values()}


def test_unauthorized_relation_never_admits_cross_generation_transit():
    f = FrontierFixture(inaccessible_bridge_bundles())
    _, result, _, _, budget = loaded(f)
    assert result.snapshot.identity_keys == (f.seeds[0].identity_key,)
    assert f.shared_identity not in result.snapshot.identity_keys
    assert f.answer_identity not in result.snapshot.identity_keys
    assert not result.snapshot.relation_groups and budget.edges == 0


def test_fresh_final_manifest_change_discards_all_candidates():
    f = FrontierFixture()
    driver = FrontierIndexFixtureDriver(f.bundles)
    original = driver.execute_read
    manifests = []

    def read(query, parameters, **kwargs):
        rows = original(query, parameters, **kwargs)
        if query == MANIFEST_BATCH_LOOKUP:
            manifests.append(True)
            if len(manifests) == 2:
                rows[0]["record"]["graph_checksum"] = "a" * 64
        return rows

    driver.execute_read = read
    with pytest.raises(TopologyLoadError):
        loaded(f, driver)
    assert len(manifests) == 2


@pytest.mark.parametrize("uncertain", [False, True])
def test_exhausted_or_uncertain_original_ledger_does_not_dispatch(uncertain):
    f = FrontierFixture()
    driver = FrontierIndexFixtureDriver(f.bundles)
    budget = f.budget(max_queries=1)
    if uncertain:
        budget.stop_uncertain()
    with pytest.raises(FrontierUnavailableV2):
        loaded(f, driver, budget=budget)
    assert budget.queries == len(driver.calls) == 0
    assert budget._uncertain is uncertain


def test_seed_only_snapshot_needs_no_unused_document_provenance():
    bundles = tuple(
        replace(
            b,
            relations=(),
            relation_semantics=(),
            evidence=(),
            entity_mentions=(),
            counts=replace(
                b.counts,
                relation_count=0,
                relation_semantics_count=0,
                evidence_count=0,
                entity_mention_count=0,
            ),
        )
        for b in bridge_bundles()
    )
    f = FrontierFixture(bundles)
    _, result, _, _, _ = loaded(f)
    assert result.snapshot.identity_keys == (f.seeds[0].identity_key,)
    assert result.evidence_candidates == ()
    assert all(
        p.scope_type.value == "collection" for p in result.snapshot.artifact_provenance
    )


def test_prior_consumed_document_read_never_substitutes_for_a_fresh_page_read():
    f = multi_document_fixture()
    f.session.initialize()
    bundle = f.bundles[0]
    generation = bundle.generation.generation_key
    origin = bundle.relations[0].source_entity_key
    copy, _ = next(f.session.copies(generation, origin))
    stream = "relations/" + origin + "/knows"
    entry = next(
        f.session.entries(generation, copy["links"]["relations"]["knows"], stream)
    )
    page = next(f.session.relation(generation, entry, stream=stream, depth=0))
    f.assembly.add_relation(page, entry, generation, origin, 0)
    document = page.relation_evidence[0].document_key
    del f.driver.raw[generation, document]
    before = f.budget.queries
    with pytest.raises(TopologyLoadError):
        next(f.session.relation(generation, entry, stream=stream, depth=0))
    assert f.budget.queries > before
