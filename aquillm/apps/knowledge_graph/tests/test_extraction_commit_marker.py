from __future__ import annotations

import uuid
from types import SimpleNamespace

ONTOLOGY_PATH = (
    __import__("pathlib").Path(__file__).parents[1] / "ontologies" / "research-v1.yaml"
)
DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")


def test_atomic_extraction_marker_rejects_partial_or_mismatched_evidence():
    from apps.knowledge_graph.extraction.pipeline import (
        extraction_commit_is_valid,
    )

    committed = SimpleNamespace(
        artifact_id=None,
        ontology_checksum="a" * 64,
        assembly_version="not-applicable",
        assembly_config_checksum="b" * 64,
        stats={
            "extraction_evidence_fingerprint": "c" * 64,
            "extraction_commit": {
                "version": 1,
                "assembly_version": "not-applicable",
                "assembly_config_checksum": "b" * 64,
                "entity_mention_count": 2,
                "relation_mention_count": 1,
            },
        },
    )
    partial = SimpleNamespace(stats={"entity_mention_count": 2})
    missing_checksum = SimpleNamespace(
        stats={"extraction_commit": committed.stats["extraction_commit"]}
    )

    class ChecksumSubclass(str):
        pass

    subclass_checksum = SimpleNamespace(
        artifact_id=None,
        ontology_checksum=ChecksumSubclass("a" * 64),
        assembly_version="not-applicable",
        assembly_config_checksum="b" * 64,
        stats=committed.stats,
    )

    assert extraction_commit_is_valid(committed, entity_count=2, relation_count=1)
    assert extraction_commit_is_valid(
        committed,
        entity_count=2,
        relation_count=1,
        evidence_fingerprint="c" * 64,
    )
    assert not extraction_commit_is_valid(
        committed,
        entity_count=2,
        relation_count=1,
        evidence_fingerprint="d" * 64,
    )
    assert not extraction_commit_is_valid(committed, entity_count=1, relation_count=1)
    assert not extraction_commit_is_valid(partial, entity_count=2, relation_count=1)
    assert not extraction_commit_is_valid(
        missing_checksum, entity_count=2, relation_count=1
    )
    assert not extraction_commit_is_valid(
        subclass_checksum, entity_count=2, relation_count=1
    )
