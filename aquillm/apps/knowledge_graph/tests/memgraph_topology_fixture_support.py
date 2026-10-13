"""Projection-only synthetic mapping; this does not claim PostgreSQL persistence."""

from dataclasses import replace
from uuid import UUID

from apps.knowledge_graph.projection.records import PrivateProjectionChunkReferenceV1
from apps.knowledge_graph.projection.serialization import private_chunk_mapping_checksum
from apps.knowledge_graph.tests.test_memgraph_projection_repository import _expected_manifest


def _stage_ready(repository, bundle):
    documents = {
        row.document_key: str(UUID(int=i))
        for i, row in enumerate(bundle.documents, 1)
    }
    mapping = tuple(
        PrivateProjectionChunkReferenceV1(
            row.chunk_key, i, documents[row.document_key], row.chunk_number,
        )
        for i, row in enumerate(sorted(bundle.chunks, key=lambda r: r.chunk_key), 1)
    )
    assert len(mapping) == bundle.counts.chunk_count > 0
    empty = _expected_manifest(bundle)
    checksum = private_chunk_mapping_checksum(mapping)
    assert checksum != private_chunk_mapping_checksum(()) == empty.private_mapping_checksum
    key = repository.opaque_generation_key(bundle.generation.generation_key)
    repository.write_staging_generation(
        bundle=bundle, private_mapping_checksum=empty.private_mapping_checksum,
        batch_size=2, timeout_seconds=5.0,
    )
    assert repository.validate_generation(expected=empty, timeout_seconds=5.0).valid is False
    assert repository.delete_generation(generation_key=key, timeout_seconds=5.0)
    expected = replace(empty, private_mapping_checksum=checksum)
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
        generation_key=key,
        validation_checksum=validation.validation_checksum,
        timeout_seconds=5.0,
    )
    return validation
