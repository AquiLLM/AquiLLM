"""Live Memgraph projection/topology acceptance, gated to the host harness."""

from __future__ import annotations

import os
from dataclasses import replace
from hashlib import sha256
from time import monotonic

import pytest

from apps.knowledge_graph.projection import runtime
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.projection.topology_snapshot import (
    build_projected_topology_snapshot,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    ready_generation_bundle_checksum,
)
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
)
from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    _expected_manifest,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import (
    _caps,
    _ready,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle

_FAMILIES = (
    "entities",
    "automatic_memberships",
    "documents",
    "chunks",
    "relation_semantics",
    "relations",
    "evidence",
    "entity_mentions",
    "artifact_provenance",
)


def _second_generation(bundle):
    generation_key = sha256(b"task22-second-generation").hexdigest()
    projection_key = sha256(b"task22-second-projection").hexdigest()

    def rekey(row):
        if hasattr(row, "generation_key"):
            return replace(row, generation_key=generation_key)
        return row

    families = {
        name: tuple(rekey(row) for row in getattr(bundle, name))
        for name in _FAMILIES
    }
    return replace(
        bundle,
        generation=replace(
            bundle.generation,
            generation_key=generation_key,
            projection_key=projection_key,
        ),
        **families,
    )


def _ready_for_document(bundle, document_key):
    ready = _ready(bundle)
    documents = tuple(
        row for row in ready.authorized_documents if row.document_key == document_key
    )
    signature = ready.authorization_context_signature
    return ReadyGenerationBundleV1(
        ready.selected_generations,
        documents,
        signature,
        ready_generation_bundle_checksum(
            ready.selected_generations,
            documents,
            signature,
        ),
    )


def _with_unrelated_document(bundle):
    document_key = sha256(b"task22-unrelated-document").hexdigest()
    document = replace(bundle.documents[0], document_key=document_key)
    chunk = replace(
        bundle.chunks[0],
        chunk_key=sha256(b"task22-unrelated-chunk").hexdigest(),
        document_key=document_key,
        chunk_number=0,
    )
    provenance = replace(
        bundle.artifact_provenance[1],
        artifact_key=sha256(b"task22-unrelated-artifact").hexdigest(),
        scope_key=document_key,
    )
    return replace(
        bundle,
        documents=tuple(
            sorted((*bundle.documents, document), key=lambda row: row.document_key)
        ),
        chunks=tuple(
            sorted(
                (*bundle.chunks, chunk),
                key=lambda row: (row.document_key, row.chunk_number, row.chunk_key),
            )
        ),
        artifact_provenance=tuple(
            sorted(
                (*bundle.artifact_provenance, provenance),
                key=lambda row: (row.scope_type, row.scope_key, row.artifact_key),
            )
        ),
        counts=replace(
            bundle.counts,
            document_count=2,
            chunk_count=2,
            artifact_provenance_count=3,
        ),
    )


def _stage_ready(repository, bundle):
    expected = _expected_manifest(bundle)
    repository.write_staging_generation(
        bundle=bundle,
        private_mapping_checksum=expected.private_mapping_checksum,
        batch_size=2,
        timeout_seconds=5.0,
    )
    stale = repository.validate_generation(
        expected=replace(expected, graph_checksum="f" * 64),
        timeout_seconds=5.0,
    )
    assert stale.valid is False
    validation = repository.validate_generation(
        expected=expected,
        timeout_seconds=5.0,
    )
    assert validation.valid is True
    repository.mark_generation_ready(
        generation_key=repository.opaque_generation_key(
            bundle.generation.generation_key
        ),
        validation_checksum=validation.validation_checksum,
        timeout_seconds=5.0,
    )
    return validation


def test_two_generation_fixture_covers_candidate_automatic_and_evidence() -> None:
    first = _bundle()
    second = _second_generation(first)

    assert first.generation.collection_key == second.generation.collection_key
    assert first.generation.generation_key != second.generation.generation_key
    assert projection_checksum(first) != projection_checksum(second)
    assert {
        row.automatic_membership_key is None for row in first.automatic_memberships
    } == {False, True}
    assert first.relation_semantics and first.evidence and first.entity_mentions


def test_selected_document_admits_evidence_and_unselected_document_does_not() -> None:
    bundle = _with_unrelated_document(_bundle())
    selected_document = bundle.evidence[0].document_key
    unselected_document = next(
        row.document_key
        for row in bundle.documents
        if row.document_key != selected_document
    )
    caps = _caps()
    seeds = (ProjectedSeedV1(bundle.entities[0].entity_key, 1.0),)

    selected = build_projected_topology_snapshot(
        ready=_ready_for_document(bundle, selected_document),
        seeds=seeds,
        caps=caps,
        bundles=(bundle,),
    )
    unselected = build_projected_topology_snapshot(
        ready=_ready_for_document(bundle, unselected_document),
        seeds=seeds,
        caps=caps,
        bundles=(bundle,),
    )

    assert selected.relation_groups and selected.mentions
    assert unselected.allowed_scope.document_keys == (unselected_document,)
    assert unselected.relation_groups == unselected.mentions == ()


@pytest.mark.container
@pytest.mark.skipif(
    os.environ.get("KG_REQUIRE_CONTAINER_TESTS") != "1",
    reason="Task22 live Memgraph proof runs only in the host harness",
)
def test_live_memgraph_projects_rebuilds_filters_and_prunes_generations() -> None:
    settings = runtime.load_projection_runtime_settings()
    repository = runtime.memgraph_projection_repository(settings)
    assert isinstance(repository, MemgraphProjectionRepository)
    repository.ensure_schema(timeout_seconds=5.0)
    first = _with_unrelated_document(_bundle())
    second = _second_generation(first)
    keys = tuple(
        repository.opaque_generation_key(bundle.generation.generation_key)
        for bundle in (first, second)
    )
    for key in keys:
        repository.delete_generation(generation_key=key, timeout_seconds=5.0)
    try:
        _stage_ready(repository, first)
        _stage_ready(repository, second)
        manifests = repository.list_generations(
            collection_key=None,
            after_generation_key=None,
            limit=10,
            timeout_seconds=5.0,
        )
        assert {row.generation_key for row in manifests} == {
            first.generation.generation_key,
            second.generation.generation_key,
        }

        selected_document = first.evidence[0].document_key
        unselected_document = next(
            row.document_key
            for row in first.documents
            if row.document_key != selected_document
        )
        caps = _caps()
        seeds = (ProjectedSeedV1(first.entities[0].entity_key, 1.0),)
        loader = MemgraphProjectedTopologyLoader(
            Neo4jProjectedTopologyQueryAdapter(repository._driver)
        )
        selected = loader.load(
            ready=_ready_for_document(first, selected_document),
            seeds=seeds,
            caps=caps,
            deadline=monotonic() + 15.0,
        )
        unselected = loader.load(
            ready=_ready_for_document(first, unselected_document),
            seeds=seeds,
            caps=caps,
            deadline=monotonic() + 15.0,
        )
        assert selected.relation_groups and selected.mentions
        assert unselected.allowed_scope.document_keys == (unselected_document,)
        assert unselected.relation_groups == unselected.mentions == ()

        for key in keys:
            repository.delete_generation(generation_key=key, timeout_seconds=5.0)
        assert repository.list_generations(
            collection_key=None,
            after_generation_key=None,
            limit=10,
            timeout_seconds=5.0,
        ) == ()
        _stage_ready(repository, first)
        rebuilt = repository.read_generation_manifest(
            generation_key=keys[0],
            timeout_seconds=5.0,
        )
        assert rebuilt.graph_checksum == projection_checksum(first)
    finally:
        for key in keys:
            repository.delete_generation(generation_key=key, timeout_seconds=5.0)
