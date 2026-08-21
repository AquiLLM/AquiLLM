"""Generation-scoped Memgraph topology relationship materialization."""

from __future__ import annotations

from dataclasses import asdict

from .records import CollectionGraphProjectionBundleV1


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
        driver.execute_write(
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
        driver.execute_write(
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
        driver.execute_write(
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
        driver.execute_write(
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
        driver.execute_write(
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
