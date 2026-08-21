"""Stream and attest every relationship incident to a projection generation."""

from __future__ import annotations

from .memgraph_edges import (
    EDGE_FAMILIES,
    TopologyEdgeAttestationV1,
    topology_edge_attestation,
    topology_edge_attestation_from_iterables,
)
from .memgraph_pagination import PAGE_SIZE

_EDGE_LIMITS = (50_000, 250_000, 250_000, 250_000, 250_000)
_MARKER_QUERY = (
    "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
    "RETURN g.topology_checksum AS topology_checksum, "
    + ", ".join(
        f"g.{family}_count AS {family}_count" for family in EDGE_FAMILIES
    )
)


def _query(source_label, relationship, target_label, cursor, returned):
    del source_label, target_label
    return (
        f"MATCH (source)-[edge:{relationship}]->(target) "
        "WHERE (source.generation_key = $generation_key"
        " OR target.generation_key = $generation_key"
        " OR edge.generation_key = $generation_key) "
        f"AND (NOT $has_cursor OR {cursor} > $cursor_key "
        f"OR ({cursor} = $cursor_key AND id(edge) > $cursor_id)) "
        f"RETURN {returned}, {cursor} AS cursor_key, id(edge) AS cursor_id "
        "ORDER BY cursor_key, cursor_id LIMIT $page_limit"
    )


_COMMON = (
    "edge.generation_key AS generation_key, "
    "source.generation_key AS source_generation_key, "
    "target.generation_key AS target_generation_key, "
)
_EDGE_QUERIES = (
    _query(
        "ProjectedEntity",
        "ENTITY_MEMBERSHIP",
        "AutomaticMembership",
        "coalesce(source.opaque_key, '')",
        _COMMON
        + "source.opaque_key AS entity_key, "
        "target.automatic_membership_key AS automatic_membership_key",
    ),
    _query(
        "ProjectedDocument",
        "DOCUMENT_CHUNK",
        "ProjectedChunk",
        "coalesce(target.opaque_key, '')",
        _COMMON
        + "source.opaque_key AS document_key, target.opaque_key AS chunk_key, "
        "edge.chunk_number AS chunk_number",
    ),
    _query(
        "ProjectedEntity",
        "PROJECTED_RELATION",
        "ProjectedEntity",
        "coalesce(edge.relation_key, '')",
        _COMMON
        + "edge.relation_key AS relation_key, edge.artifact_key AS artifact_key, "
        "source.opaque_key AS source_entity_key, edge.relation_type AS relation_type, "
        "target.opaque_key AS target_entity_key, edge.direction AS direction",
    ),
    _query(
        "ProjectedRelation",
        "RELATION_EVIDENCE",
        "ProjectedChunk",
        "coalesce(edge.evidence_key, '')",
        _COMMON
        + "edge.evidence_key AS evidence_key, source.opaque_key AS relation_key, "
        "edge.relation_mention_key AS relation_mention_key, "
        "target.opaque_key AS chunk_key, edge.document_key AS document_key, "
        "edge.chunk_number AS chunk_number, edge.confidence AS confidence, "
        "edge.provenance_key AS provenance_key, "
        "edge.semantic_signature AS semantic_signature, "
        "edge.head_mention_key AS head_mention_key, "
        "edge.tail_mention_key AS tail_mention_key, edge.orientation AS orientation",
    ),
    _query(
        "ProjectedEntity",
        "ENTITY_MENTION",
        "ProjectedChunk",
        "coalesce(edge.mention_key, '')",
        _COMMON
        + "edge.mention_key AS mention_key, edge.provenance_key AS provenance_key, "
        "source.opaque_key AS entity_key, target.opaque_key AS chunk_key, "
        "edge.document_key AS document_key, edge.chunk_number AS chunk_number, "
        "edge.confidence AS confidence",
    ),
)


def _mapping(row):
    if type(row) is not dict:
        raise ValueError("Memgraph topology edge row is invalid")
    value = row.get("edge", row)
    if not isinstance(value, dict):
        raise ValueError("Memgraph topology edge properties are invalid")
    return {
        key: item
        for key, item in value.items()
        if key not in {"cursor_key", "cursor_id"}
    }


def _read_marker(driver, generation_key, timeout_seconds):
    rows = driver.execute_read(
        _MARKER_QUERY,
        {"generation_key": generation_key},
        timeout_seconds=timeout_seconds,
        max_records=1,
    )
    return _mapping(rows[0]) if len(rows) == 1 else {}


def _stream_edge_family(
    driver, generation_key, query, expected_count, maximum, timeout_seconds
):
    if type(expected_count) is not int or not 0 <= expected_count <= maximum:
        raise ValueError("Memgraph topology edge count is invalid")
    count, cursor_key, cursor_id = 0, "", -1
    has_cursor = False
    while True:
        page_limit = min(PAGE_SIZE, max(1, expected_count - count + 1))
        rows = driver.execute_read(
            query,
            {
                "generation_key": generation_key,
                "has_cursor": has_cursor,
                "cursor_key": cursor_key,
                "cursor_id": cursor_id,
                "page_limit": page_limit,
            },
            timeout_seconds=timeout_seconds,
            max_records=page_limit,
        )
        if type(rows) is not tuple or len(rows) > page_limit:
            raise ValueError("Memgraph topology edge page is invalid")
        for row in rows:
            count += 1
            if count > expected_count:
                raise ValueError("Memgraph topology edge count drifted")
            yield _mapping(row)
        if len(rows) < page_limit:
            break
        last = rows[-1]
        next_key, next_id = last.get("cursor_key"), last.get("cursor_id")
        if type(next_key) is not str or type(next_id) is not int or next_id < 0:
            raise ValueError("Memgraph topology edge cursor is invalid")
        if has_cursor and (next_key, next_id) <= (cursor_key, cursor_id):
            raise ValueError("Memgraph topology edge pagination made no progress")
        has_cursor, cursor_key, cursor_id = True, next_key, next_id
    if count != expected_count:
        raise ValueError("Memgraph topology edge count drifted")


def validated_topology_attestation(
    driver, generation_key, *, timeout_seconds: float
) -> TopologyEdgeAttestationV1 | None:
    marker = _read_marker(driver, generation_key, timeout_seconds)
    checksum = marker.get("topology_checksum")
    counts = tuple(marker.get(f"{family}_count") for family in EDGE_FAMILIES)
    if (
        type(checksum) is not str
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        return None
    observed = topology_edge_attestation_from_iterables(
        tuple(
            _stream_edge_family(
                driver,
                generation_key,
                query,
                count,
                maximum,
                timeout_seconds,
            )
            for query, count, maximum in zip(
                _EDGE_QUERIES, counts, _EDGE_LIMITS, strict=True
            )
        )
    )
    if observed.checksum == checksum and observed.counts == counts:
        return observed
    return None


def validate_topology_edges(driver, bundle, *, timeout_seconds: float) -> bool:
    observed = validated_topology_attestation(
        driver, bundle.generation.generation_key, timeout_seconds=timeout_seconds
    )
    return observed == topology_edge_attestation(bundle)


def validate_topology_marker(driver, generation_key, *, timeout_seconds: float) -> bool:
    return (
        validated_topology_attestation(
            driver, generation_key, timeout_seconds=timeout_seconds
        )
        is not None
    )


__all__ = [
    "validate_topology_edges",
    "validate_topology_marker",
    "validated_topology_attestation",
]
