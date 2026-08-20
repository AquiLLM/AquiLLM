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
        "ProjectedRelationEvidenceV1 ProjectedArtifactProvenanceV1 "
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
        "ProjectionFailureStateV1": "state failure_code attempt_count",
        "ProjectionCountsV1": (
            "entity_count automatic_membership_count document_count chunk_count "
            "relation_count evidence_count artifact_provenance_count"
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


def _records(value: object, kind: type[object], name: str, key_fields: str) -> None:
    if type(value) is not tuple or any(type(row) is not kind for row in value):
        raise TypeError(f"{name} must be an exact tuple of {kind.__name__}")
    keys = tuple(
        tuple(getattr(row, field) for field in key_fields.split()) for row in value
    )
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} must be unique")
    if keys != tuple(sorted(keys)):
        raise ValueError(f"{name} must be canonically sorted")


def _validate_bundle(bundle: object) -> None:
    from . import records

    if type(bundle.generation) is not records.ProjectionGenerationMarkerV1:
        raise TypeError("generation must be ProjectionGenerationMarkerV1")
    if type(bundle.counts) is not records.ProjectionCountsV1:
        raise TypeError("counts must be ProjectionCountsV1")
    specs = (
        "entities:ProjectedEntityV1:entity_key|automatic_memberships:"
        "AutomaticCanonicalMembershipV1:entity_key|documents:"
        "ProjectedDocumentMembershipV1:document_key|chunks:"
        "ProjectedChunkMembershipV1:document_key chunk_number chunk_key|relations:"
        "ProjectedPhysicalRelationV1:relation_key|evidence:"
        "ProjectedRelationEvidenceV1:evidence_key|artifact_provenance:"
        "ProjectedArtifactProvenanceV1:scope_type scope_key artifact_key"
    ).split("|")
    for spec in specs:
        name, kind_name, key_fields = spec.split(":")
        _records(getattr(bundle, name), getattr(records, kind_name), name, key_fields)
    marker = bundle.generation
    entities = {row.entity_key for row in bundle.entities}
    documents = {row.document_key for row in bundle.documents}
    chunks = {
        row.chunk_key: (row.document_key, row.chunk_number) for row in bundle.chunks
    }
    relations = {row.relation_key: row for row in bundle.relations}
    if any(
        (row.generation_key, row.artifact_key, row.collection_key)
        != (marker.generation_key, marker.artifact_key, marker.collection_key)
        for row in bundle.entities
    ):
        raise ValueError("entity generation/schema/projection/key versions disagree")
    memberships = bundle.automatic_memberships
    if tuple(row.entity_key for row in memberships) != tuple(
        row.entity_key for row in bundle.entities
    ):
        raise ValueError("automatic membership completeness is broken")
    if any(row.generation_key != marker.generation_key for row in bundle.documents):
        raise ValueError("document generation agreement is broken")
    if any(row.document_key not in documents for row in bundle.chunks):
        raise ValueError("chunk document closure is broken")
    chunk_keys = tuple(row.chunk_key for row in bundle.chunks)
    coordinates = tuple((row.document_key, row.chunk_number) for row in bundle.chunks)
    if any(len(set(items)) != len(items) for items in (chunk_keys, coordinates)):
        raise ValueError("chunk keys and coordinates must each be unique")
    if any(
        row.source_entity_key not in entities
        or row.target_entity_key not in entities
        or row.source_entity_key == row.target_entity_key
        or row.artifact_key != marker.artifact_key
        for row in bundle.relations
    ):
        raise ValueError("relation endpoint/self-loop closure is broken")
    semantic_relations = tuple(
        (r.artifact_key, r.source_entity_key, r.relation_type, r.target_entity_key)
        for r in bundle.relations
    )
    if len(set(semantic_relations)) != len(semantic_relations):
        raise ValueError("relation semantic tuples must be unique")
    provenance = bundle.artifact_provenance
    collection_rows = tuple(row for row in provenance if row.scope_type == "collection")
    document_rows = tuple(row for row in provenance if row.scope_type == "document")
    if len(collection_rows) != 1 or (
        collection_rows[0].artifact_key,
        collection_rows[0].scope_key,
        collection_rows[0].collection_key,
    ) != (marker.artifact_key, marker.collection_key, marker.collection_key):
        raise ValueError("collection provenance closure is broken")
    documents_by_scope = {row.scope_key: row for row in document_rows}
    if (
        len(documents_by_scope) != len(document_rows)
        or set(documents_by_scope) != documents
        or any(row.collection_key != marker.collection_key for row in provenance)
    ):
        raise ValueError("document provenance closure is broken")
    shared = (
        "resolver_version resolution_config_checksum ontology_version "
        "ontology_checksum filter_policy_version filter_policy_checksum "
        "extractor_version orchestration_version"
    ).split()
    if any(len({getattr(row, name) for row in provenance}) != 1 for name in shared):
        raise ValueError("shared provenance identity is inconsistent")
    collection_provenance = collection_rows[0]
    if any(
        row.decision_checksum != marker.membership_checksum
        or row.resolver_version != collection_provenance.resolver_version
        or row.resolution_config_checksum
        != collection_provenance.resolution_config_checksum
        for row in memberships
    ):
        raise ValueError("automatic membership coherence is broken")
    if any(
        row.relation_key not in relations
        or row.chunk_key not in chunks
        or row.source_document_key != row.document_key
        or chunks.get(row.chunk_key) != (row.document_key, row.chunk_number)
        or row.relation_type != relations[row.relation_key].relation_type
        or row.document_key not in documents_by_scope
        or row.artifact_key != documents_by_scope[row.document_key].artifact_key
        or row.ontology_checksum != collection_provenance.ontology_checksum
        or row.assembly_config_checksum
        != collection_provenance.assembly_config_checksum
        for row in bundle.evidence
    ):
        raise ValueError("evidence coherence is broken")
    pairs = (
        "entity_count:entities automatic_membership_count:automatic_memberships "
        "document_count:documents chunk_count:chunks relation_count:relations "
        "evidence_count:evidence artifact_provenance_count:artifact_provenance"
    ).split()
    actual = tuple(getattr(bundle.counts, pair.split(":")[0]) for pair in pairs)
    expected = tuple(len(getattr(bundle, pair.split(":")[1])) for pair in pairs)
    if actual != expected:
        raise ValueError("projection count agreement is broken")


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
