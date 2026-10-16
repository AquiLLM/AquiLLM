"""Existing canonical fixtures with an exact consumed-source ranking context."""

from dataclasses import replace
from types import SimpleNamespace

from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2
from apps.knowledge_graph.projection.frontier_snapshot_v2 import FrontierAssemblyV2
from apps.knowledge_graph.retrieval.frontier_ranking_validation_v2 import (
    CONSUMED_POLICY,
    FrontierRankingContextV2,
)
from apps.knowledge_graph.retrieval.graph_query_contracts import GraphRelationRequestV2
from apps.knowledge_graph.retrieval.topology.frontier_acquisition_v2 import (
    acquire_frontier,
)
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
    key,
)
from apps.knowledge_graph.tests.frontier_scope_window_support import mixed_bundle
from apps.knowledge_graph.tests.test_projected_topology_adapter import _caps, _ready


def configure_consumed(fixture):
    fixture.request = GraphRelationRequestV2(
        fixture.ready.selected_generations[0].ontology_checksum,
        (("knows", 1.0),),
        "relation-request-v2",
        key("query"),
    )
    fixture.session = FrontierSessionV2(
        fixture.driver,
        group=fixture.group,
        request=fixture.request,
        budget=fixture.budget,
        deadline=fixture.budget.deadline,
        acquisition_policy=CONSUMED_POLICY,
    )
    context = FrontierRankingContextV2(
        fixture.group.group_checksum,
        *(
            getattr(fixture.budget, name)
            for name in (
                "max_identities",
                "max_edges",
                "max_source_rows",
                "max_queries",
                "max_wire_bytes",
            )
        ),
        CONSUMED_POLICY,
    )
    fixture.caps = _caps()
    fixture.assembly = FrontierAssemblyV2(
        fixture.session,
        SimpleNamespace(representation=lambda _g, _d: None),
        fixture.caps,
        context,
    )
    return fixture


def multi_document_fixture(*, mention_only=False):
    fixture = GenerationFixture(1, 4)
    bundle = mixed_bundle(fixture.bundles[0], 0)
    source = {p.scope_key: p for p in bundle.artifact_provenance}
    chunks, evidence, mentions = [], [], []
    for i, document in enumerate(bundle.documents):
        chunk = replace(
            bundle.chunks[0],
            chunk_key=key(f"chunk:{i}"),
            document_key=document.document_key,
        )
        chunks.append(chunk)
        evidence.append(
            replace(
                bundle.evidence[0],
                evidence_key=key(f"evidence:{i}"),
                provenance_key=key(f"provenance:{i}"),
                chunk_key=chunk.chunk_key,
                document_key=document.document_key,
                source_document_key=document.document_key,
                artifact_key=source[document.document_key].artifact_key,
                confidence=0.9 - i / 10,
            )
        )
        mentions.append(
            replace(
                bundle.entity_mentions[0],
                mention_key=key(f"mention:{i}"),
                provenance_key=key(f"mention-proof:{i}"),
                chunk_key=chunk.chunk_key,
                document_key=document.document_key,
                confidence=0.9 - i / 10,
            )
        )
    bundle = replace(
        bundle,
        chunks=tuple(chunks),
        evidence=tuple(sorted(evidence, key=lambda r: r.evidence_key))
        if not mention_only
        else (),
        entity_mentions=tuple(
            sorted(
                mentions, key=lambda r: (r.entity_key, r.provenance_key, r.mention_key)
            )
        ),
        relations=() if mention_only else bundle.relations,
        relation_semantics=() if mention_only else bundle.relation_semantics,
        counts=replace(
            bundle.counts,
            chunk_count=4,
            evidence_count=0 if mention_only else 4,
            entity_mention_count=4,
            relation_count=0 if mention_only else 1,
            relation_semantics_count=0 if mention_only else 1,
        ),
    )
    from apps.knowledge_graph.tests.frontier_generation_batch_support import (
        GenerationDriver,
    )

    fixture.bundles = (bundle,)
    fixture.ready = _ready(bundle)
    fixture.group.scope.ready = fixture.ready
    fixture.driver = GenerationDriver(fixture.bundles)
    return configure_consumed(fixture)


def acquire_fixture(fixture):
    from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1

    bundle = fixture.bundles[0]
    seed = ProjectedSeedV1(bundle.entity_mentions[0].entity_key, 1.0)
    fixture.session.initialize()
    acquire_frontier(
        fixture.session, fixture.assembly, (seed,), fixture.caps, fixture.budget
    )
    fixture.session.final_attestation()
    return (seed,)
