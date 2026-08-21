"""Generation-scoped Memgraph topology relationship materialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

from .memgraph_driver import MemgraphWriteSummaryV1
from .records import CollectionGraphProjectionBundleV1

EDGE_FAMILIES = (
    "entity_membership",
    "document_chunk",
    "projected_relation",
    "relation_evidence",
    "entity_mention",
)


@dataclass(frozen=True, slots=True)
class TopologyEdgeAttestationV1:
    checksum: str
    counts: tuple[int, int, int, int, int]


def topology_edge_rows(bundle: CollectionGraphProjectionBundleV1):
    generation_key = bundle.generation.generation_key
    directions = {
        (row.artifact_key, row.relation_type): row.direction
        for row in bundle.relation_semantics
    }
    memberships = tuple(
        {
            "generation_key": generation_key,
            "entity_key": row.entity_key,
            "automatic_membership_key": row.automatic_membership_key,
        }
        for row in bundle.automatic_memberships
    )
    chunks = tuple(
        {"generation_key": generation_key, **asdict(row)} for row in bundle.chunks
    )
    relations = tuple(
        {
            "generation_key": generation_key,
            **asdict(row),
            "direction": directions[(row.artifact_key, row.relation_type)],
        }
        for row in bundle.relations
    )
    evidence = tuple(
        {
            "generation_key": generation_key,
            "evidence_key": row.evidence_key,
            "relation_key": row.relation_key,
            "relation_mention_key": row.relation_mention_key,
            "chunk_key": row.chunk_key,
            "document_key": row.document_key,
            "chunk_number": row.chunk_number,
            "confidence": row.confidence,
            "provenance_key": row.provenance_key,
            "semantic_signature": row.semantic_signature,
            "head_mention_key": row.head_mention_key,
            "tail_mention_key": row.tail_mention_key,
            "orientation": row.orientation,
        }
        for row in bundle.evidence
    )
    mentions = tuple(
        {"generation_key": generation_key, **asdict(row)}
        for row in bundle.entity_mentions
    )
    return memberships, chunks, relations, evidence, mentions


def _canonical_rows(rows) -> bytes:
    normalized = [
        {
            "family": family,
            "rows": [
                {
                    key: value.hex() if type(value) is float else value
                    for key, value in sorted(row.items())
                }
                for row in records
            ],
        }
        for family, records in zip(EDGE_FAMILIES, rows, strict=True)
    ]
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def topology_edge_attestation(
    bundle: CollectionGraphProjectionBundleV1,
) -> TopologyEdgeAttestationV1:
    rows = topology_edge_rows(bundle)
    counts = tuple(len(family) for family in rows)
    return TopologyEdgeAttestationV1(sha256(_canonical_rows(rows)).hexdigest(), counts)


def topology_edge_attestation_from_rows(rows) -> TopologyEdgeAttestationV1:
    if type(rows) is not tuple or len(rows) != len(EDGE_FAMILIES):
        raise ValueError("Memgraph topology edge families are invalid")
    counts = tuple(len(family) for family in rows)
    return TopologyEdgeAttestationV1(sha256(_canonical_rows(rows)).hexdigest(), counts)


def _write(driver, cypher, parameters, *, timeout_seconds: float) -> None:
    summary = driver.execute_write(
        cypher, parameters, timeout_seconds=timeout_seconds
    )
    if type(summary) is not MemgraphWriteSummaryV1:
        raise TypeError("Memgraph topology write summary is invalid")


def write_topology_edges(
    driver,
    bundle: CollectionGraphProjectionBundleV1,
    *,
    timeout_seconds: float,
) -> None:
    """Materialize opaque topology edges after all endpoint nodes exist."""
    generation_key = bundle.generation.generation_key
    semantics = {
        (row.artifact_key, row.relation_type): row.direction
        for row in bundle.relation_semantics
    }
    for row in bundle.chunks:
        _write(
            driver,
            "MATCH (d:ProjectedDocument {generation_key:$generation_key, "
            "opaque_key:$document_key}) "
            "MATCH (c:ProjectedChunk {generation_key:$generation_key, "
            "opaque_key:$chunk_key}) "
            "MERGE (d)-[edge:DOCUMENT_CHUNK {generation_key:$generation_key, "
            "chunk_key:$chunk_key}]->(c) "
            "SET edge.document_key=$document_key, edge.chunk_number=$chunk_number",
            {"generation_key": generation_key, **asdict(row)},
            timeout_seconds=timeout_seconds,
        )
    for row in bundle.automatic_memberships:
        _write(
            driver,
            "MATCH (entity:ProjectedEntity {generation_key:$generation_key, "
            "opaque_key:$entity_key}) "
            "MATCH (membership:AutomaticMembership {generation_key:$generation_key, "
            "opaque_key:$entity_key}) "
            "MERGE (entity)-[:ENTITY_MEMBERSHIP {generation_key:$generation_key, "
            "entity_key:$entity_key}]->(membership)",
            {"generation_key": generation_key, "entity_key": row.entity_key},
            timeout_seconds=timeout_seconds,
        )
    for row in bundle.relations:
        direction = semantics.get((row.artifact_key, row.relation_type))
        if direction is None:
            raise ValueError("physical relation has no projected semantics")
        _write(
            driver,
            "MATCH (source:ProjectedEntity {generation_key:$generation_key, "
            "opaque_key:$source_entity_key}) "
            "MATCH (target:ProjectedEntity {generation_key:$generation_key, "
            "opaque_key:$target_entity_key}) "
            "MERGE (source)-[edge:PROJECTED_RELATION {generation_key:$generation_key, "
            "relation_key:$relation_key}]->(target) "
            "SET edge.artifact_key=$artifact_key, edge.relation_type=$relation_type, "
            "edge.direction=$direction",
            {"generation_key": generation_key, "direction": direction, **asdict(row)},
            timeout_seconds=timeout_seconds,
        )
    for row in bundle.entity_mentions:
        _write(
            driver,
            "MATCH (entity:ProjectedEntity {generation_key:$generation_key, "
            "opaque_key:$entity_key}) "
            "MATCH (chunk:ProjectedChunk {generation_key:$generation_key, "
            "opaque_key:$chunk_key}) "
            "MERGE (entity)-[edge:ENTITY_MENTION {generation_key:$generation_key, "
            "mention_key:$mention_key}]->(chunk) "
            "SET edge.document_key=$document_key, edge.chunk_number=$chunk_number, "
            "edge.confidence=$confidence, edge.provenance_key=$provenance_key",
            {"generation_key": generation_key, **asdict(row)},
            timeout_seconds=timeout_seconds,
        )
    for row in bundle.evidence:
        _write(
            driver,
            "MATCH (relation:ProjectedRelation {generation_key:$generation_key, "
            "opaque_key:$relation_key}) "
            "MATCH (chunk:ProjectedChunk {generation_key:$generation_key, "
            "opaque_key:$chunk_key}) "
            "MERGE (relation)-[edge:RELATION_EVIDENCE {generation_key:$generation_key, "
            "evidence_key:$evidence_key}]->(chunk) "
            "SET edge.document_key=$document_key, edge.chunk_number=$chunk_number, "
            "edge.confidence=$confidence, edge.provenance_key=$provenance_key, "
            "edge.semantic_signature=$semantic_signature, "
            "edge.relation_mention_key=$relation_mention_key, "
            "edge.head_mention_key=$head_mention_key, "
            "edge.tail_mention_key=$tail_mention_key, edge.orientation=$orientation",
            {"generation_key": generation_key, **asdict(row)},
            timeout_seconds=timeout_seconds,
        )
