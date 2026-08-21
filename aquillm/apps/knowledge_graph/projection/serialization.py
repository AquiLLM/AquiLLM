from __future__ import annotations

import json
from dataclasses import fields
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from uuid import UUID

_MAX_COUNT = 2**31 - 1


def canonical_projection_bytes(value: object) -> bytes:
    """Serialize only the closed provider-neutral projection contract."""
    provider_types, bundle_type = _provider_types()
    if type(value) is tuple:
        _validate_top_level_records(value, provider_types)
    elif type(value) is not bundle_type and type(value) not in provider_types:
        raise TypeError("value is not a supported provider-neutral projection type")
    encoded = _encode(value, provider_types)
    return json.dumps(
        encoded,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def projection_checksum(value: object) -> str:
    return sha256(canonical_projection_bytes(value)).hexdigest()


def private_chunk_mapping_checksum(value: object) -> str:
    from .records import PrivateProjectionChunkReferenceV1

    if type(value) is not tuple or any(
        type(row) is not PrivateProjectionChunkReferenceV1 for row in value
    ):
        raise TypeError("private chunk mapping must be an exact record tuple")
    columns = (
        tuple(row.projection_chunk_key for row in value),
        tuple(row.integer_chunk_pk for row in value),
        tuple((row.document_uuid, row.chunk_number) for row in value),
    )
    if any(len(set(column)) != len(column) for column in columns):
        raise ValueError("private chunk mapping must be unique")
    keys = tuple(zip(*columns, strict=True))
    if keys != tuple(sorted(keys)):
        raise ValueError("private chunk mapping must be canonically sorted")
    payload = [
        {
            "chunk_number": row.chunk_number,
            "document_uuid": row.document_uuid,
            "integer_chunk_pk": row.integer_chunk_pk,
            "projection_chunk_key": row.projection_chunk_key,
        }
        for row in value
    ]
    data = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256(data).hexdigest()


def _provider_types() -> tuple[frozenset[type[object]], type[object]]:
    from . import records

    names = (
        "ProjectionGenerationMarkerV1 ProjectedEntityV1 "
        "AutomaticCanonicalMembershipV1 ProjectedDocumentMembershipV1 "
        "ProjectedChunkMembershipV1 ProjectedPhysicalRelationV1 "
        "ProjectedRelationSemanticsV1 ProjectedRelationEvidenceV1 "
        "ProjectedEntityMentionEvidenceV1 ProjectedArtifactProvenanceV1 "
        "ProjectionCountsV1 ProjectionGenerationManifestV1 ProjectionLeaseV1 "
        "ProjectionFailureStateV1"
    ).split()
    bundle_type = records.CollectionGraphProjectionBundleV1
    return frozenset(getattr(records, name) for name in names), bundle_type


def _validate_top_level_records(
    value: tuple[object, ...], provider_types: frozenset[type[object]]
) -> None:
    if not value or type(value[0]) not in provider_types:
        raise TypeError("tuple does not contain a supported projection record type")
    kind = type(value[0])
    if any(type(row) is not kind for row in value):
        raise TypeError("projection record tuples must contain one exact record type")
    special = {
        "ProjectedChunkMembershipV1": "document_key chunk_number chunk_key",
        "ProjectedArtifactProvenanceV1": "scope_type scope_key artifact_key",
        "ProjectedRelationSemanticsV1": "artifact_key relation_type semantics_key",
        "ProjectedEntityMentionEvidenceV1": "entity_key provenance_key mention_key",
        "ProjectionFailureStateV1": "state failure_code attempt_count",
        "ProjectionCountsV1": (
            "entity_count automatic_membership_count document_count chunk_count "
            "relation_semantics_count relation_count evidence_count "
            "entity_mention_count artifact_provenance_count"
        ),
    }
    key_fields = special.get(kind.__name__, fields(value[0])[0].name).split()
    keys = tuple(tuple(getattr(row, name) for name in key_fields) for row in value)
    if len(set(keys)) != len(keys):
        raise ValueError("projection record tuple must be unique")
    if keys != tuple(sorted(keys)):
        raise ValueError("projection record tuple must be canonically sorted")


def _encode(value: object, provider_types: frozenset[type[object]]) -> object:
    from .records import CollectionGraphProjectionBundleV1

    if type(value) in provider_types | {CollectionGraphProjectionBundleV1}:
        return {
            field.name: _encode(getattr(value, field.name), provider_types)
            for field in fields(value)
        }
    if type(value) is tuple:
        return [_encode(item, provider_types) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    if type(value) is datetime:
        if value.tzinfo is not UTC:
            raise ValueError("datetime must be exact UTC")
        return value.isoformat().replace("+00:00", "Z")
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("numeric value must be finite")
        return value.hex()
    if value is None or type(value) in {str, int, bool}:
        return value
    raise TypeError("record contains an unsupported canonical value")


def _key(value: object, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in str")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal key")


def _token(value: object, name: str, maximum: int = 512) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in str")
    if not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded canonical token")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error


def _count(
    value: object, name: str, minimum: int = 0, maximum: int = _MAX_COUNT
) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its bounded range")


def _uuid(value: object, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in str")
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical UUID") from error
    if value != str(parsed):
        raise ValueError(f"{name} must be a canonical lowercase UUID")
