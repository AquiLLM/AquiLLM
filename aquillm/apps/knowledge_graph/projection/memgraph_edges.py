"""Generation-scoped Memgraph topology relationship materialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import islice

from .memgraph_driver import MemgraphWriteSummaryV1
from .records import CollectionGraphProjectionBundleV1

EDGE_FAMILIES = (
    "entity_membership",
    "document_chunk",
    "projected_relation",
    "relation_evidence",
    "entity_mention",
)
EDGE_ORDER_FIELDS = (
    "entity_key",
    "chunk_key",
    "relation_key",
    "evidence_key",
    "mention_key",
)
_STAGING_GUARD = (
    "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
    "WHERE g.state IN ['staging','building'] WITH g "
)
# Keep structural UNWIND queries and their scalar parameters small even when a
# caller requests the public 5,000-row maximum. No family rows are discarded.
_MAX_BOLT_BATCH_ROWS = 128
_MAX_BOLT_BATCH_PARAMETERS = 4_096
_MAX_BOLT_BATCH_QUERY_BYTES = 131_072


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
            "source_generation_key": generation_key,
            "target_generation_key": generation_key,
            "entity_key": row.entity_key,
            "automatic_membership_key": row.automatic_membership_key,
        }
        for row in bundle.automatic_memberships
    )
    chunks = tuple(
        {
            "generation_key": generation_key,
            "source_generation_key": generation_key,
            "target_generation_key": generation_key,
            **asdict(row),
        }
        for row in bundle.chunks
    )
    relations = tuple(
        {
            "generation_key": generation_key,
            "source_generation_key": generation_key,
            "target_generation_key": generation_key,
            **asdict(row),
            "direction": directions[(row.artifact_key, row.relation_type)],
        }
        for row in bundle.relations
    )
    evidence = tuple(
        {
            "generation_key": generation_key,
            "source_generation_key": generation_key,
            "target_generation_key": generation_key,
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
        {
            "generation_key": generation_key,
            "source_generation_key": generation_key,
            "target_generation_key": generation_key,
            **asdict(row),
        }
        for row in bundle.entity_mentions
    )
    return memberships, chunks, relations, evidence, mentions


def _normalized_row(row) -> dict:
    return {
        key: value.hex() if type(value) is float else value
        for key, value in sorted(row.items())
    }


def _ordered(rows):
    return tuple(
        tuple(sorted(records, key=lambda row: row[order_field]))
        for records, order_field in zip(rows, EDGE_ORDER_FIELDS, strict=True)
    )


def topology_edge_attestation_from_iterables(rows) -> TopologyEdgeAttestationV1:
    if type(rows) is not tuple or len(rows) != len(EDGE_FAMILIES):
        raise ValueError("Memgraph topology edge families are invalid")
    digest = sha256()
    counts = []
    digest.update(b"[")
    for family_index, (family, records) in enumerate(
        zip(EDGE_FAMILIES, rows, strict=True)
    ):
        if family_index:
            digest.update(b",")
        digest.update(b'{"family":')
        digest.update(json.dumps(family).encode("utf-8"))
        digest.update(b',"rows":[')
        count = 0
        for row in records:
            if count:
                digest.update(b",")
            digest.update(
                json.dumps(
                    _normalized_row(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            count += 1
        counts.append(count)
        digest.update(b"]}")
    digest.update(b"]")
    return TopologyEdgeAttestationV1(digest.hexdigest(), tuple(counts))


def topology_edge_attestation(
    bundle: CollectionGraphProjectionBundleV1,
) -> TopologyEdgeAttestationV1:
    rows = _ordered(topology_edge_rows(bundle))
    return topology_edge_attestation_from_iterables(rows)


def topology_edge_attestation_from_rows(rows) -> TopologyEdgeAttestationV1:
    return topology_edge_attestation_from_iterables(_ordered(rows))


def write_parameterized_batches(
    driver,
    cypher,
    rows,
    *,
    generation_key: str,
    batch_size: int,
    timeout_seconds: float,
) -> None:
    """Write each bounded slice atomically through the scalar-only driver.

    Map field names come from internal record schemas, never record values.
    Flattening into scalar parameters preserves the driver's validation and
    keeps retries inside its managed transaction callback.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 5_000:
        raise ValueError("batch_size must be an integer in 1..5000")
    iterator = iter(rows)
    while batch := tuple(islice(iterator, min(batch_size, _MAX_BOLT_BATCH_ROWS))):
        fields = tuple(sorted(set(batch[0]) - {"generation_key"}))
        if not fields or any(
            type(name) is not str or not name.isascii() or not name.isidentifier()
            for name in fields
        ):
            raise ValueError("Memgraph batch fields must be internal identifiers")
        parameters = {"generation_key": generation_key}
        maps = []
        for index, row in enumerate(batch):
            if set(row) - {"generation_key"} != set(fields):
                raise ValueError("Memgraph batch record fields differ")
            if row.get("generation_key", generation_key) != generation_key:
                raise ValueError("Memgraph batch generation differs")
            assignments = []
            for name in fields:
                parameter = f"row{index}_{name}"
                parameters[parameter] = row[name]
                assignments.append(f"{name}:${parameter}")
            maps.append("{" + ",".join(assignments) + "}")
        query = _STAGING_GUARD + "UNWIND [" + ",".join(maps) + "] AS row " + cypher
        if (
            len(parameters) > _MAX_BOLT_BATCH_PARAMETERS
            or len(query.encode("utf-8")) > _MAX_BOLT_BATCH_QUERY_BYTES
        ):
            raise ValueError("Memgraph batch exceeds bounded query capacity")
        summary = driver.execute_write(
            query, parameters, timeout_seconds=timeout_seconds
        )
        if type(summary) is not MemgraphWriteSummaryV1:
            raise TypeError("Memgraph batch write summary is invalid")


def write_topology_edges(
    driver,
    bundle: CollectionGraphProjectionBundleV1,
    *,
    timeout_seconds: float,
    batch_size: int = 128,
) -> None:
    """Materialize opaque topology edges after all endpoint nodes exist."""
    generation_key = bundle.generation.generation_key
    semantics = {
        (row.artifact_key, row.relation_type): row.direction
        for row in bundle.relation_semantics
    }
    write_parameterized_batches(
        driver,
        "MATCH (d:ProjectedDocument:ProjectedRecord {generation_key:$generation_key, "
        "opaque_key:row.document_key}) "
        "MATCH (c:ProjectedChunk:ProjectedRecord {generation_key:$generation_key, "
        "opaque_key:row.chunk_key}) "
        "MERGE (d)-[edge:DOCUMENT_CHUNK {generation_key:$generation_key, "
        "chunk_key:row.chunk_key}]->(c) "
        "SET edge.document_key=row.document_key, edge.chunk_number=row.chunk_number",
        (asdict(row) for row in bundle.chunks),
        generation_key=generation_key,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
    )
    write_parameterized_batches(
        driver,
        "MATCH (entity:ProjectedEntity:ProjectedRecord "
        "{generation_key:$generation_key, "
        "opaque_key:row.entity_key}) "
        "MATCH (membership:AutomaticMembership:ProjectedRecord "
        "{generation_key:$generation_key, "
        "opaque_key:row.entity_key}) "
        "MERGE (entity)-[:ENTITY_MEMBERSHIP {generation_key:$generation_key, "
        "entity_key:row.entity_key}]->(membership)",
        ({"entity_key": row.entity_key} for row in bundle.automatic_memberships),
        generation_key=generation_key,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
    )

    def relation_rows():
        for row in bundle.relations:
            direction = semantics.get((row.artifact_key, row.relation_type))
            if direction is None:
                raise ValueError("physical relation has no projected semantics")
            yield {"direction": direction, **asdict(row)}

    write_parameterized_batches(
        driver,
        "MATCH (source:ProjectedEntity:ProjectedRecord "
        "{generation_key:$generation_key, "
        "opaque_key:row.source_entity_key}) "
        "MATCH (target:ProjectedEntity:ProjectedRecord "
        "{generation_key:$generation_key, "
        "opaque_key:row.target_entity_key}) "
        "MERGE (source)-[edge:PROJECTED_RELATION {generation_key:$generation_key, "
        "relation_key:row.relation_key}]->(target) "
        "SET edge.artifact_key=row.artifact_key, edge.relation_type=row.relation_type, "
        "edge.direction=row.direction",
        relation_rows(),
        generation_key=generation_key,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
    )
    write_parameterized_batches(
        driver,
        "MATCH (entity:ProjectedEntity:ProjectedRecord "
        "{generation_key:$generation_key, "
        "opaque_key:row.entity_key}) "
        "MATCH (chunk:ProjectedChunk:ProjectedRecord {generation_key:$generation_key, "
        "opaque_key:row.chunk_key}) "
        "MERGE (entity)-[edge:ENTITY_MENTION {generation_key:$generation_key, "
        "mention_key:row.mention_key}]->(chunk) "
        "SET edge.document_key=row.document_key, edge.chunk_number=row.chunk_number, "
        "edge.confidence=row.confidence, edge.provenance_key=row.provenance_key",
        (asdict(row) for row in bundle.entity_mentions),
        generation_key=generation_key,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
    )
    write_parameterized_batches(
        driver,
        "MATCH (relation:ProjectedRelation:ProjectedRecord "
        "{generation_key:$generation_key, "
        "opaque_key:row.relation_key}) "
        "MATCH (chunk:ProjectedChunk:ProjectedRecord {generation_key:$generation_key, "
        "opaque_key:row.chunk_key}) "
        "MERGE (relation)-[edge:RELATION_EVIDENCE {generation_key:$generation_key, "
        "evidence_key:row.evidence_key}]->(chunk) "
        "SET edge.document_key=row.document_key, edge.chunk_number=row.chunk_number, "
        "edge.confidence=row.confidence, edge.provenance_key=row.provenance_key, "
        "edge.semantic_signature=row.semantic_signature, "
        "edge.relation_mention_key=row.relation_mention_key, "
        "edge.head_mention_key=row.head_mention_key, "
        "edge.tail_mention_key=row.tail_mention_key, edge.orientation=row.orientation",
        (asdict(row) for row in bundle.evidence),
        generation_key=generation_key,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
    )
