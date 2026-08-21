"""Bounded streaming validation for full Memgraph projection generations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from .memgraph_pagination import (
    FAMILY_IDENTITY_FIELDS,
    PAGE_SIZE,
    advance_cursor_parameters,
    canonical_record_key,
    full_family_page_query,
    initial_cursor_parameters,
)
from .memgraph_records import FAMILIES, _dto, _properties
from .records import (
    ProjectionCountsV1,
    ProjectionGenerationMarkerV1,
)
from .serialization import canonical_projection_bytes

_FAMILY_LIMITS = {
    "ProjectedEntity": 50_000,
    "AutomaticMembership": 50_000,
    "ProjectedDocument": 10_000,
    "ProjectedChunk": 250_000,
    "ProjectedRelationSemantics": 10_000,
    "ProjectedRelation": 250_000,
    "ProjectedEvidence": 250_000,
    "ProjectedEntityMention": 250_000,
    "ArtifactProvenance": 10_001,
}
_COUNT_FIELDS = {
    "ProjectedEntity": "entity_count",
    "AutomaticMembership": "automatic_membership_count",
    "ProjectedDocument": "document_count",
    "ProjectedChunk": "chunk_count",
    "ProjectedRelationSemantics": "relation_semantics_count",
    "ProjectedRelation": "relation_count",
    "ProjectedEvidence": "evidence_count",
    "ProjectedEntityMention": "entity_mention_count",
    "ArtifactProvenance": "artifact_provenance_count",
}
_BUNDLE_FIELDS = (
    ("artifact_provenance", "ArtifactProvenance"),
    ("automatic_memberships", "AutomaticMembership"),
    ("chunks", "ProjectedChunk"),
    ("counts", None),
    ("documents", "ProjectedDocument"),
    ("entities", "ProjectedEntity"),
    ("entity_mentions", "ProjectedEntityMention"),
    ("evidence", "ProjectedEvidence"),
    ("generation", None),
    ("relation_semantics", "ProjectedRelationSemantics"),
    ("relations", "ProjectedRelation"),
)


@dataclass(frozen=True, slots=True)
class StreamedProjectionValidationV1:
    marker: ProjectionGenerationMarkerV1
    checksum: str
    counts: ProjectionCountsV1


def validate_projection_count_limits(counts: ProjectionCountsV1) -> None:
    if type(counts) is not ProjectionCountsV1:
        raise TypeError("counts must be exact ProjectionCountsV1")
    for label, maximum in _FAMILY_LIMITS.items():
        if getattr(counts, _COUNT_FIELDS[label]) > maximum:
            raise ValueError("projection count exceeds supported validation limit")


def stream_family_records(
    driver,
    *,
    generation_key: str,
    label: str,
    expected_count: int,
    timeout_seconds: float,
):
    kinds = dict(FAMILIES)
    if label not in kinds or type(expected_count) is not int or expected_count < 0:
        raise ValueError("Memgraph streaming family request is invalid")
    if expected_count > _FAMILY_LIMITS[label]:
        raise ValueError("projection count exceeds supported validation limit")
    parameters = {
        "generation_key": generation_key,
        **initial_cursor_parameters(label),
    }
    query = full_family_page_query(label)
    count = 0
    previous_key = None
    while True:
        page_limit = min(PAGE_SIZE, max(1, expected_count - count + 1))
        page_parameters = {**parameters, "page_limit": page_limit}
        rows = driver.execute_read(
            query,
            page_parameters,
            timeout_seconds=timeout_seconds,
            max_records=page_limit,
        )
        if type(rows) is not tuple or len(rows) > page_limit:
            raise ValueError("Memgraph projection page is invalid")
        last_properties = None
        for row in rows:
            properties = _properties(row, "record")
            record = _dto(kinds[label], properties)
            identity = FAMILY_IDENTITY_FIELDS[label]
            if properties.get("opaque_key") != getattr(record, identity):
                raise ValueError("Memgraph projection opaque identity drifted")
            key = canonical_record_key(label, record)
            if previous_key is not None and key <= previous_key:
                raise ValueError("Memgraph projection page order is invalid")
            previous_key, last_properties = key, properties
            count += 1
            if count > expected_count:
                raise ValueError("Memgraph projection family count drifted")
            yield record
        if len(rows) < page_limit:
            break
        if last_properties is None:
            raise ValueError("Memgraph projection pagination made no progress")
        parameters.update(
            advance_cursor_parameters(label, last_properties, rows[-1].get("cursor_id"))
        )
    if count != expected_count:
        raise ValueError("Memgraph projection family count drifted")


def _read_marker(driver, generation_key: str, timeout_seconds: float):
    rows = driver.execute_read(
        "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
        "RETURN g AS record",
        {"generation_key": generation_key},
        timeout_seconds=timeout_seconds,
        max_records=1,
    )
    if len(rows) != 1:
        raise ValueError("generation marker is missing")
    return _dto(ProjectionGenerationMarkerV1, _properties(rows[0], "record"))


def stream_projection_validation(
    driver,
    *,
    generation_key: str,
    expected_counts: ProjectionCountsV1,
    timeout_seconds: float,
) -> StreamedProjectionValidationV1:
    validate_projection_count_limits(expected_counts)
    marker = _read_marker(driver, generation_key, timeout_seconds)
    digest = sha256()
    digest.update(b"{")
    for index, (field, label) in enumerate(_BUNDLE_FIELDS):
        if index:
            digest.update(b",")
        digest.update(json.dumps(field).encode("utf-8") + b":")
        if label is None:
            value = expected_counts if field == "counts" else marker
            digest.update(canonical_projection_bytes(value))
            continue
        digest.update(b"[")
        expected = getattr(expected_counts, _COUNT_FIELDS[label])
        for row_index, row in enumerate(
            stream_family_records(
                driver,
                generation_key=generation_key,
                label=label,
                expected_count=expected,
                timeout_seconds=timeout_seconds,
            )
        ):
            if row_index:
                digest.update(b",")
            digest.update(canonical_projection_bytes(row))
        digest.update(b"]")
    digest.update(b"}")
    return StreamedProjectionValidationV1(
        marker, digest.hexdigest(), expected_counts
    )


__all__ = [
    "StreamedProjectionValidationV1",
    "stream_family_records",
    "stream_projection_validation",
    "validate_projection_count_limits",
]
