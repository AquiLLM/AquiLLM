from __future__ import annotations

from dataclasses import fields

from .memgraph_pagination import (
    FAMILY_IDENTITY_FIELDS,
    PAGE_SIZE,
    advance_cursor_parameters,
    canonical_record_key,
    initial_cursor_parameters,
)
from .records import (
    AutomaticCanonicalMembershipV1,
    CollectionGraphProjectionBundleV1,
    ProjectedArtifactProvenanceV1,
    ProjectedChunkMembershipV1,
    ProjectedDocumentMembershipV1,
    ProjectedEntityMentionEvidenceV1,
    ProjectedEntityV1,
    ProjectedPhysicalRelationV1,
    ProjectedRelationEvidenceV1,
    ProjectedRelationSemanticsV1,
    ProjectionCountsV1,
    ProjectionGenerationManifestV1,
    ProjectionGenerationMarkerV1,
    ProjectionLifecycleState,
)
from .topology_cypher import bounded_family_query

FAMILIES = (
    ("ProjectedEntity", ProjectedEntityV1),
    ("AutomaticMembership", AutomaticCanonicalMembershipV1),
    ("ProjectedDocument", ProjectedDocumentMembershipV1),
    ("ProjectedChunk", ProjectedChunkMembershipV1),
    ("ProjectedRelationSemantics", ProjectedRelationSemanticsV1),
    ("ProjectedRelation", ProjectedPhysicalRelationV1),
    ("ProjectedEvidence", ProjectedRelationEvidenceV1),
    ("ProjectedEntityMention", ProjectedEntityMentionEvidenceV1),
    ("ArtifactProvenance", ProjectedArtifactProvenanceV1),
)

_ORDER_FIELDS = (
    "entity_key",
    "entity_key",
    "document_key",
    "document_key chunk_number chunk_key",
    "artifact_key relation_type semantics_key",
    "relation_key",
    "evidence_key",
    "entity_key provenance_key mention_key",
    "scope_type scope_key artifact_key",
)


def _read_topology_family(
    driver,
    *,
    query,
    parameters,
    label,
    kind,
    maximum,
    timeout,
    reject_full_pages,
):
    cursor = initial_cursor_parameters(label)
    loaded = []
    previous_key = None
    while True:
        remaining = maximum - len(loaded)
        if not reject_full_pages and remaining == 0:
            break
        page_limit = min(PAGE_SIZE, max(1, remaining + (1 if reject_full_pages else 0)))
        page_parameters = {**parameters, **cursor, "page_limit": page_limit}
        rows = driver.execute_read(
            query,
            page_parameters,
            timeout_seconds=timeout,
            max_records=page_limit,
        )
        if type(rows) is not tuple or len(rows) > page_limit:
            raise ValueError("bounded Memgraph family page is invalid")
        last_properties = None
        for row in rows:
            properties = _properties(row, "record")
            decoded = _dto(kind, properties)
            identity = FAMILY_IDENTITY_FIELDS[label]
            if properties.get("opaque_key") != getattr(decoded, identity):
                raise ValueError("bounded Memgraph family identity drifted")
            key = canonical_record_key(label, decoded)
            if previous_key is not None and key <= previous_key:
                raise ValueError("bounded Memgraph family order drifted")
            previous_key, last_properties = key, properties
            loaded.append(decoded)
            if reject_full_pages and len(loaded) > maximum:
                raise ValueError("bounded Memgraph family read was truncated")
        if len(rows) < page_limit:
            break
        if last_properties is None:
            raise ValueError("bounded Memgraph pagination made no progress")
        cursor = advance_cursor_parameters(
            label, last_properties, rows[-1].get("cursor_id")
        )
    return tuple(loaded)


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
        "relation_semantics_count",
        "relation_count",
        "evidence_count",
        "entity_mention_count",
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
    driver,
    *,
    generation_key: str,
    maxima: tuple[int, ...],
    timeout: float,
    reject_full_pages: bool = False,
    topology_parameters: dict[str, object] | None = None,
):
    maximum_cap = 50_000 if topology_parameters is not None else 5_000
    if len(maxima) != len(FAMILIES) or any(
        type(value) is not int or not 1 <= value <= maximum_cap for value in maxima
    ):
        raise ValueError("Memgraph projection read maxima are invalid")
    parameters = {"generation_key": generation_key}
    if topology_parameters is not None:
        parameters.update(topology_parameters)
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
    for (label, kind), maximum, order_fields in zip(
        FAMILIES, maxima, _ORDER_FIELDS, strict=True
    ):
        if topology_parameters is not None:
            query = bounded_family_query(label)
        else:
            query = (
                f"MATCH (n:{label} {{generation_key:$generation_key}}) "
                "RETURN n AS record ORDER BY n.opaque_key"
            )
        if topology_parameters is not None:
            decoded = _read_topology_family(
                driver,
                query=query,
                parameters=parameters,
                label=label,
                kind=kind,
                maximum=maximum,
                timeout=timeout,
                reject_full_pages=reject_full_pages,
            )
        else:
            rows = driver.execute_read(
                query,
                parameters,
                timeout_seconds=timeout,
                max_records=maximum,
            )
            decoded = tuple(_dto(kind, _properties(row, "record")) for row in rows)
        fields = order_fields.split()
        loaded.append(
            tuple(
                sorted(
                    decoded,
                    key=lambda row: tuple(getattr(row, field) for field in fields),
                )
            )
        )
    (
        entities,
        memberships,
        documents,
        chunks,
        relation_semantics,
        relations,
        evidence,
        entity_mentions,
        provenance,
    ) = loaded
    counts = ProjectionCountsV1(
        len(entities),
        len(memberships),
        len(documents),
        len(chunks),
        len(relation_semantics),
        len(relations),
        len(evidence),
        len(entity_mentions),
        len(provenance),
    )
    return CollectionGraphProjectionBundleV1(
        marker,
        entities,
        memberships,
        documents,
        chunks,
        relation_semantics,
        relations,
        evidence,
        entity_mentions,
        provenance,
        counts,
    )
