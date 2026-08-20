from __future__ import annotations

from dataclasses import fields

from .records import (
    AutomaticCanonicalMembershipV1,
    CollectionGraphProjectionBundleV1,
    ProjectedArtifactProvenanceV1,
    ProjectedChunkMembershipV1,
    ProjectedDocumentMembershipV1,
    ProjectedEntityV1,
    ProjectedPhysicalRelationV1,
    ProjectedRelationEvidenceV1,
    ProjectionCountsV1,
    ProjectionGenerationManifestV1,
    ProjectionGenerationMarkerV1,
    ProjectionLifecycleState,
)

FAMILIES = (
    ("ProjectedEntity", ProjectedEntityV1),
    ("AutomaticMembership", AutomaticCanonicalMembershipV1),
    ("ProjectedDocument", ProjectedDocumentMembershipV1),
    ("ProjectedChunk", ProjectedChunkMembershipV1),
    ("ProjectedRelation", ProjectedPhysicalRelationV1),
    ("ProjectedEvidence", ProjectedRelationEvidenceV1),
    ("ArtifactProvenance", ProjectedArtifactProvenanceV1),
)


def _properties(row: object, name: str) -> dict:
    if type(row) is not dict:
        raise ValueError(f"Memgraph {name} row is invalid")
    value = row.get(name, row)
    try:
        return dict(value)
    except (TypeError, ValueError):
        raise ValueError(f"Memgraph {name} properties are invalid") from None


def _dto(kind, properties: dict):
    names = tuple(field.name for field in fields(kind))
    try:
        values = {name: properties[name] for name in names}
    except KeyError:
        raise ValueError("Memgraph projection record is incomplete") from None
    return kind(**values)


def manifest_from_row(row: object) -> ProjectionGenerationManifestV1:
    properties = _properties(row, "manifest")
    state = properties.get("state")
    lifecycle = (
        ProjectionLifecycleState.BUILDING
        if state == "staging"
        else ProjectionLifecycleState(state)
    )
    names = (
        "entity_count",
        "automatic_membership_count",
        "document_count",
        "chunk_count",
        "relation_count",
        "evidence_count",
        "artifact_provenance_count",
    )
    try:
        counts = ProjectionCountsV1(*(properties[name] for name in names))
        return ProjectionGenerationManifestV1(
            properties["generation_key"],
            properties["schema_version"],
            properties["projection_version"],
            properties["identifier_key_version"],
            properties["graph_checksum"],
            properties["snapshot_checksum"],
            properties["private_mapping_checksum"],
            counts,
            lifecycle,
        )
    except KeyError:
        raise ValueError("Memgraph manifest row is incomplete") from None


def read_bundle(
    driver, *, generation_key: str, maxima: tuple[int, ...], timeout: float
):
    if len(maxima) != len(FAMILIES) or any(
        type(value) is not int or not 1 <= value <= 5_000 for value in maxima
    ):
        raise ValueError("Memgraph projection read maxima are invalid")
    parameters = {"generation_key": generation_key}
    marker_rows = driver.execute_read(
        "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
        "RETURN g AS record",
        parameters,
        timeout_seconds=timeout,
        max_records=1,
    )
    if len(marker_rows) != 1:
        raise ValueError("generation marker is missing")
    marker = _dto(ProjectionGenerationMarkerV1, _properties(marker_rows[0], "record"))
    loaded = []
    for (label, kind), maximum in zip(FAMILIES, maxima, strict=True):
        rows = driver.execute_read(
            f"MATCH (n:{label} {{generation_key:$generation_key}}) "
            "RETURN n AS record ORDER BY n.opaque_key",
            parameters,
            timeout_seconds=timeout,
            max_records=maximum,
        )
        loaded.append(tuple(_dto(kind, _properties(row, "record")) for row in rows))
    entities, memberships, documents, chunks, relations, evidence, provenance = loaded
    counts = ProjectionCountsV1(
        len(entities),
        len(memberships),
        len(documents),
        len(chunks),
        len(relations),
        len(evidence),
        len(provenance),
    )
    return CollectionGraphProjectionBundleV1(
        marker,
        entities,
        memberships,
        documents,
        chunks,
        relations,
        evidence,
        provenance,
        counts,
    )
