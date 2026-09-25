from dataclasses import replace

import pytest

from apps.knowledge_graph.tests.test_projection_records import _bundle


def test_evidence_assembly_uses_collection_not_document_provenance() -> None:
    bundle = _bundle()
    evidence = bundle.evidence[0]
    collection, document = bundle.artifact_provenance
    assert collection.assembly_config_checksum == evidence.assembly_config_checksum
    assert evidence.assembly_config_checksum != document.assembly_config_checksum


def test_bundle_rejects_chunk_key_and_coordinate_conflicts_independently() -> None:
    bundle = _bundle()
    original = bundle.chunks[0]
    for duplicate in (
        replace(original, chunk_number=3),
        replace(original, chunk_key="e" * 64),
    ):
        rows = tuple(
            sorted(
                (original, duplicate),
                key=lambda row: (row.document_key, row.chunk_number, row.chunk_key),
            )
        )
        with pytest.raises(ValueError, match="chunk"):
            replace(bundle, chunks=rows, counts=replace(bundle.counts, chunk_count=2))
