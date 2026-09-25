from __future__ import annotations

from dataclasses import asdict

from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.projection.records import (
    CollectionGraphProjectionBundleV1,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    TopologyCapsV1,
)
from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    _bundle as _raw_bundle,
)
from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    _FakeDriver,
    _family_rows,
)


def test_generation_records_are_bounded_and_endpoint_closed() -> None:
    bundle = CollectionGraphProjectionBundleV1(**_raw_bundle())
    driver = _FakeDriver()
    driver.read_results = [
        ({"record": asdict(bundle.generation)},),
        _family_rows(bundle, bundle.entities, "entity_key"),
        _family_rows(bundle, bundle.automatic_memberships, "entity_key"),
        _family_rows(bundle, bundle.documents, "document_key"),
        _family_rows(bundle, bundle.chunks, "chunk_key"),
        _family_rows(bundle, bundle.relation_semantics, "semantics_key"),
        _family_rows(bundle, bundle.relations, "relation_key"),
        _family_rows(bundle, bundle.evidence, "evidence_key"),
        _family_rows(bundle, bundle.entity_mentions, "mention_key"),
        _family_rows(bundle, bundle.artifact_provenance, "scope_key"),
    ]
    caps = TopologyCapsV1(HybridBranchKind.DIRECT, 2, 1, 2, 2, 2)

    observed = MemgraphProjectionRepository(driver).read_generation_records(
        generation_key=MemgraphProjectionRepository.opaque_generation_key("1" * 64),
        caps=caps,
        timeout_seconds=1.0,
    )

    assert observed == bundle
    assert all(call[3] <= 2 for call in driver.reads)
