"""Versioned consumed-source closure through genuine canonical frontier fixtures."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.frontier_adapter_v2 import (
    Neo4jCanonicalFrontierAdapterV2,
)
from apps.knowledge_graph.projection.frontier_snapshot_v2 import frontier_config
from apps.knowledge_graph.retrieval.projected_relation_ranking_v2 import (
    rank_relations_v2,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_loader_v2 import (
    canonical_frontier_loader_v2,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
)
from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
    FrontierIndexFixtureDriver,
)

POLICY = (
    "global-priority-quality-keyset-v3/nonexhaustive/authorized-transit/"
    "consumed-source-closure"
)
TAGS = (
    "canonical_frontier_v3",
    "frontier_transition_v2",
    "frontier_evidence_v3",
    "frontier_seed_v2",
)


def loaded(fixture=None, driver=None, *, policy=POLICY, budget=None):
    fixture = fixture or FrontierFixture()
    driver = driver or FrontierIndexFixtureDriver(fixture.bundles)
    loader = canonical_frontier_loader_v2(
        selection=fixture.selection,
        codec=fixture.codec,
        adapter=Neo4jCanonicalFrontierAdapterV2(driver, acquisition_policy=policy),
        acquisition_policy=policy,
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
    context = loader.ranking_context(result)
    config = frontier_config(
        fixture.caps, fixture.group.scope.ready.selected_generations[0].resolver_version
    )
    if result.snapshot.identity_keys:
        rank_relations_v2(
            snapshot=result.snapshot,
            seeds=fixture.seeds,
            request=fixture.request,
            config=config,
            frontier_context=context,
            deadline_check=budget.check_active,
        )
    return fixture, result, context, config, budget


def test_unused_document_defect_stays_legacy_eager_but_new_initialization_ignores_it():
    fixture = GenerationFixture(generations=1, documents=2)
    document = fixture.ready.authorized_documents[-1].document_key
    generation = fixture.ready.selected_generations[0].generation_key
    del fixture.driver.raw[generation, document]
    with pytest.raises(TopologyLoadError):
        fixture.session.initialize()
    fixture = GenerationFixture(generations=1, documents=2)
    del fixture.driver.raw[generation, document]
    from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2

    fixture.session = FrontierSessionV2(
        fixture.driver,
        group=fixture.group,
        request=fixture.session.request,
        budget=fixture.budget,
        deadline=fixture.budget.deadline,
        acquisition_policy=POLICY,
    )
    fixture.session.initialize()
    assert {
        r.scope_key
        for rows in fixture.session.scope_rows.values()
        for (label, _), r in rows.items()
        if label == "ArtifactProvenance"
    } == {fixture.ready.selected_generations[0].collection_key}
    fixture.session.final_attestation()


def test_new_snapshot_constructs_and_ranks_with_full_authority_consumed_proof():
    f, result, context, _, _ = loaded()
    assert result.evidence_candidates and result.snapshot.relation_groups
    assert context.acquisition_policy == POLICY
    assert (
        tuple(
            getattr(result.snapshot.algorithm, n)
            for n in (
                "algorithm_version",
                "transition_version",
                "evidence_version",
                "seed_version",
            )
        )
        == TAGS
    )
    assert result.snapshot.allowed_scope.document_keys == tuple(
        sorted(d.document_key for d in f.group.scope.ready.authorized_documents)
    )
    assert result.authority_group_checksum == f.group.group_checksum
    assert result.coverage.exhaustive is False


@pytest.mark.parametrize(
    "family", ["ProjectedDocument", "ProjectedChunk", "ArtifactProvenance"]
)
def test_identical_missing_consumed_proof_rejects(family):
    f = FrontierFixture()
    driver = FrontierIndexFixtureDriver(f.bundles)
    bundle = f.bundles[0]
    key = (
        bundle.documents[0].document_key
        if family != "ProjectedChunk"
        else bundle.chunks[0].chunk_key
    )
    generation = bundle.generation.generation_key
    driver.raw[generation, key] = [
        row for row in driver.raw[generation, key] if family not in row["labels"]
    ]
    with pytest.raises(TopologyLoadError):
        loaded(f, driver)


def test_empty_new_snapshot_keeps_collections_and_full_allowed_documents():
    f = FrontierFixture()
    f.seeds = (replace(f.seeds[0], identity_key="f" * 64),)
    _, result, _, _, _ = loaded(f)
    assert result.snapshot.identity_keys == ()
    assert result.evidence_candidates == ()
    assert len(result.snapshot.artifact_provenance) == len(f.bundles)
    assert all(
        p.scope_type.value == "collection" for p in result.snapshot.artifact_provenance
    )


def test_legacy_signature_cannot_be_retagged_to_consumed_version():
    from apps.knowledge_graph.retrieval.frontier_ranking_validation_v2 import (
        POLICY as LEGACY,
    )

    f, result, _, config, _ = loaded(policy=LEGACY)
    from apps.knowledge_graph.retrieval.frontier_ranking_validation_v2 import (
        FrontierRankingContextV2,
    )

    context = FrontierRankingContextV2(
        f.group.group_checksum, 200, 1000, 5000, 64, 1048576, POLICY
    )
    retagged = replace(
        result.snapshot,
        algorithm=replace(
            result.snapshot.algorithm,
            algorithm_version=TAGS[0],
            evidence_version=TAGS[2],
        ),
    )
    with pytest.raises(ValueError, match="signature"):
        rank_relations_v2(
            snapshot=retagged,
            seeds=f.seeds,
            request=f.request,
            config=config,
            frontier_context=context,
            deadline_check=lambda: None,
        )


def test_same_unused_document_defect_only_rejects_when_it_is_consumed():
    from apps.knowledge_graph.retrieval.frontier_ranking_validation_v2 import (
        POLICY as LEGACY,
    )
    from apps.knowledge_graph.tests.frontier_fixture_support import bridge_bundles

    first, second = bridge_bundles()
    second = replace(
        second,
        automatic_memberships=tuple(
            replace(m, automatic_membership_key=None)
            for m in second.automatic_memberships
        ),
    )
    f = FrontierFixture((first, second))
    driver = FrontierIndexFixtureDriver(f.bundles)
    bundle = f.bundles[1]
    del driver.raw[bundle.generation.generation_key, bundle.documents[0].document_key]
    with pytest.raises(TopologyLoadError):
        loaded(f, driver, policy=LEGACY)
    _, result, _, _, _ = loaded(f, driver)
    assert len(result.snapshot.allowed_scope.document_keys) == 2
    assert (
        len(
            [
                p
                for p in result.snapshot.artifact_provenance
                if p.scope_type.value == "document"
            ]
        )
        == 1
    )
    f.seeds = (replace(f.seeds[0], identity_key=bundle.entity_mentions[0].entity_key),)
    with pytest.raises(TopologyLoadError):
        loaded(f, driver)


@pytest.mark.parametrize("kind", ["unknown", "cross-policy"])
def test_unknown_and_cross_policy_loader_selection_rejects(kind):
    from apps.knowledge_graph.retrieval.frontier_ranking_validation_v2 import (
        POLICY as LEGACY,
    )

    f = FrontierFixture()
    driver = FrontierIndexFixtureDriver(f.bundles)
    if kind == "unknown":
        with pytest.raises(ValueError):
            loaded(f, driver, policy="consumed-but-unversioned")
    else:
        loader = canonical_frontier_loader_v2(
            selection=f.selection,
            codec=f.codec,
            adapter=Neo4jCanonicalFrontierAdapterV2(driver, acquisition_policy=LEGACY),
            acquisition_policy=POLICY,
        )
        with pytest.raises(ValueError, match="session/context"):
            loader.load(
                group=f.group,
                seeds=f.seeds,
                request=f.request,
                caps=f.caps,
                budget=f.budget(),
                deadline=f.deadline,
            )
        assert not driver.calls
