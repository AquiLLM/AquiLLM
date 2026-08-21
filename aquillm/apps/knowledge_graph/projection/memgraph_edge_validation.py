"""Read back and attest generation-scoped Memgraph topology relationships."""

from __future__ import annotations

from .memgraph_edges import (
    EDGE_FAMILIES,
    topology_edge_attestation,
    topology_edge_attestation_from_rows,
    topology_edge_rows,
)

_MARKER_QUERY = (
    "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
    "RETURN g.topology_checksum AS topology_checksum, "
    + ", ".join(
        f"g.{family}_count AS {family}_count" for family in EDGE_FAMILIES
    )
)

_EDGE_QUERIES = (
    "MATCH (entity:ProjectedEntity {generation_key:$generation_key})"
    "-[edge:ENTITY_MEMBERSHIP {generation_key:$generation_key}]->"
    "(membership:AutomaticMembership {generation_key:$generation_key}) "
    "RETURN edge.generation_key AS generation_key, "
    "entity.opaque_key AS entity_key, "
    "membership.automatic_membership_key AS automatic_membership_key "
    "ORDER BY entity_key",
    "MATCH (document:ProjectedDocument {generation_key:$generation_key})"
    "-[edge:DOCUMENT_CHUNK {generation_key:$generation_key}]->"
    "(chunk:ProjectedChunk {generation_key:$generation_key}) "
    "RETURN edge.generation_key AS generation_key, "
    "document.opaque_key AS document_key, chunk.opaque_key AS chunk_key, "
    "edge.chunk_number AS chunk_number "
    "ORDER BY document_key, chunk_number, chunk_key",
    "MATCH (source:ProjectedEntity {generation_key:$generation_key})"
    "-[edge:PROJECTED_RELATION {generation_key:$generation_key}]->"
    "(target:ProjectedEntity {generation_key:$generation_key}) "
    "RETURN edge.generation_key AS generation_key, "
    "edge.relation_key AS relation_key, edge.artifact_key AS artifact_key, "
    "source.opaque_key AS source_entity_key, edge.relation_type AS relation_type, "
    "target.opaque_key AS target_entity_key, edge.direction AS direction "
    "ORDER BY relation_key",
    "MATCH (relation:ProjectedRelation {generation_key:$generation_key})"
    "-[edge:RELATION_EVIDENCE {generation_key:$generation_key}]->"
    "(chunk:ProjectedChunk {generation_key:$generation_key}) "
    "RETURN edge.generation_key AS generation_key, "
    "edge.evidence_key AS evidence_key, relation.opaque_key AS relation_key, "
    "edge.relation_mention_key AS relation_mention_key, "
    "chunk.opaque_key AS chunk_key, edge.document_key AS document_key, "
    "edge.chunk_number AS chunk_number, edge.confidence AS confidence, "
    "edge.provenance_key AS provenance_key, "
    "edge.semantic_signature AS semantic_signature, "
    "edge.head_mention_key AS head_mention_key, "
    "edge.tail_mention_key AS tail_mention_key, edge.orientation AS orientation "
    "ORDER BY evidence_key",
    "MATCH (entity:ProjectedEntity {generation_key:$generation_key})"
    "-[edge:ENTITY_MENTION {generation_key:$generation_key}]->"
    "(chunk:ProjectedChunk {generation_key:$generation_key}) "
    "RETURN edge.generation_key AS generation_key, "
    "edge.mention_key AS mention_key, edge.provenance_key AS provenance_key, "
    "entity.opaque_key AS entity_key, chunk.opaque_key AS chunk_key, "
    "edge.document_key AS document_key, edge.chunk_number AS chunk_number, "
    "edge.confidence AS confidence "
    "ORDER BY entity_key, provenance_key, mention_key",
)


def _mapping(row):
    if type(row) is not dict:
        raise ValueError("Memgraph topology edge row is invalid")
    value = row.get("edge", row)
    if not isinstance(value, dict):
        raise ValueError("Memgraph topology edge properties are invalid")
    return dict(value)


def _read_marker(driver, generation_key, timeout_seconds):
    parameters = {"generation_key": generation_key}
    marker_rows = driver.execute_read(
        _MARKER_QUERY,
        parameters,
        timeout_seconds=timeout_seconds,
        max_records=1,
    )
    return _mapping(marker_rows[0]) if len(marker_rows) == 1 else {}


def _read_edges(driver, generation_key, counts, timeout_seconds):
    parameters = {"generation_key": generation_key}
    observed = []
    for query, count in zip(_EDGE_QUERIES, counts, strict=True):
        if type(count) is not int or not 0 <= count <= 4_999:
            raise ValueError("Memgraph topology edge count is invalid")
        rows = driver.execute_read(
            query,
            parameters,
            timeout_seconds=timeout_seconds,
            max_records=max(1, count + 1),
        )
        family = tuple(_mapping(row) for row in rows)
        observed.append(family)
    return tuple(observed)


def _marker(attestation):
    return {
        "topology_checksum": attestation.checksum,
        **{
            f"{family}_count": count
            for family, count in zip(
                EDGE_FAMILIES, attestation.counts, strict=True
            )
        },
    }


def validate_topology_edges(driver, bundle, *, timeout_seconds: float) -> bool:
    generation_key = bundle.generation.generation_key
    expected_rows = topology_edge_rows(bundle)
    expected = topology_edge_attestation(bundle)
    if _read_marker(driver, generation_key, timeout_seconds) != _marker(expected):
        return False
    observed = _read_edges(
        driver, generation_key, expected.counts, timeout_seconds
    )
    return (
        observed == expected_rows
        and topology_edge_attestation_from_rows(observed) == expected
    )


def validate_topology_marker(driver, generation_key, *, timeout_seconds: float) -> bool:
    marker = _read_marker(driver, generation_key, timeout_seconds)
    checksum = marker.get("topology_checksum")
    counts = tuple(marker.get(f"{family}_count") for family in EDGE_FAMILIES)
    if (
        type(checksum) is not str
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        return False
    observed = _read_edges(driver, generation_key, counts, timeout_seconds)
    attestation = topology_edge_attestation_from_rows(observed)
    return attestation.checksum == checksum and attestation.counts == counts


__all__ = ["validate_topology_edges", "validate_topology_marker"]
