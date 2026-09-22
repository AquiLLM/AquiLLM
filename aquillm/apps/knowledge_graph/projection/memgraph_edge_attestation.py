"""Deterministic topology edge rows and attestations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

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
