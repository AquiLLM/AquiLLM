from __future__ import annotations

from dataclasses import replace

from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    TopologyCapsV1,
)
from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    _FakeDriver,
    _record_reads,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle


def test_generation_readback_canonically_sorts_semantic_record_families() -> None:
    original = _bundle()
    relation = replace(
        original.relations[0], relation_key="f" * 64, relation_type="likes"
    )
    semantics = replace(
        original.relation_semantics[0],
        semantics_key="0" * 64,
        relation_type="likes",
    )
    evidence = replace(
        original.evidence[0],
        evidence_key="f" * 64,
        relation_key=relation.relation_key,
        relation_type="likes",
    )
    mention = replace(
        original.entity_mentions[0],
        mention_key="f" * 64,
        provenance_key="0" * 64,
    )
    counts = replace(
        original.counts,
        relation_semantics_count=2,
        relation_count=2,
        evidence_count=2,
        entity_mention_count=2,
    )
    expected = replace(
        original,
        relation_semantics=(original.relation_semantics[0], semantics),
        relations=(original.relations[0], relation),
        evidence=(original.evidence[0], evidence),
        entity_mentions=(mention, original.entity_mentions[0]),
        counts=counts,
    )
    reads = _record_reads(expected)
    reads[5] = tuple(reversed(reads[5]))
    reads[8] = tuple(reversed(reads[8]))
    driver = _FakeDriver()
    driver.read_results = reads

    observed = MemgraphProjectionRepository(driver).read_generation_records(
        generation_key=MemgraphProjectionRepository.opaque_generation_key("1" * 64),
        caps=TopologyCapsV1(HybridBranchKind.DIRECT, 2, 1, 2, 2, 2),
        timeout_seconds=1.0,
    )

    assert observed == expected
