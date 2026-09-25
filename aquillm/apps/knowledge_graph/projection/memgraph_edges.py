"""Generation-scoped Memgraph topology relationship materialization."""

from __future__ import annotations

from dataclasses import asdict
from itertools import islice

from .memgraph_driver import MemgraphWriteSummaryV1
from .memgraph_edge_attestation import (
    EDGE_FAMILIES as EDGE_FAMILIES,
)
from .memgraph_edge_attestation import (
    EDGE_ORDER_FIELDS as EDGE_ORDER_FIELDS,
)
from .memgraph_edge_attestation import (
    TopologyEdgeAttestationV1 as TopologyEdgeAttestationV1,
)
from .memgraph_edge_attestation import (
    topology_edge_attestation as topology_edge_attestation,
)
from .memgraph_edge_attestation import (
    topology_edge_attestation_from_iterables as topology_edge_attestation_from_iterables,  # noqa: E501
)
from .memgraph_edge_attestation import (
    topology_edge_attestation_from_rows as topology_edge_attestation_from_rows,
)
from .memgraph_edge_attestation import (
    topology_edge_rows as topology_edge_rows,
)
from .records import CollectionGraphProjectionBundleV1

_STAGING_GUARD = (
    "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
    "WHERE g.state IN ['staging','building'] WITH g "
)
# Family generation indexes serve reads; batch identity lookups must use both
# properties to avoid rescanning a whole generation for each UNWIND row.
_BATCH_IDENTITY_INDEX = "USING INDEX :ProjectedRecord(generation_key, opaque_key) "
# Keep structural UNWIND queries and their scalar parameters small even when a
# caller requests the public 5,000-row maximum. No family rows are discarded.
_MAX_BOLT_BATCH_ROWS = 128
_MAX_BOLT_BATCH_PARAMETERS = 4_096
_MAX_BOLT_BATCH_QUERY_BYTES = 131_072


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
        query = (
            _BATCH_IDENTITY_INDEX
            + _STAGING_GUARD
            + "UNWIND ["
            + ",".join(maps)
            + "] AS row "
            + cypher
        )
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
