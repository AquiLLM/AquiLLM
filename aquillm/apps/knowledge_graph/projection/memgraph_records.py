from __future__ import annotations

from dataclasses import fields

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

_ENTITY_TRAVERSAL = (
    "MATCH (membership:AutomaticMembership {generation_key:$generation_key}) "
    "WHERE membership.automatic_membership_key IN split($seed_keys_csv, ',') "
    "OR membership.entity_key IN split($seed_keys_csv, ',') "
    "MATCH (seed:ProjectedEntity {generation_key:$generation_key})"
    "-[:ENTITY_MEMBERSHIP]->(membership) "
    "MATCH p=(seed)"
    "-[:PROJECTED_RELATION*0..2]-(n:ProjectedEntity "
    "{generation_key:$generation_key}) "
    "WHERE seed.opaque_key IN split($seed_keys_csv, ',') "
    "AND length(p) <= $max_depth "
    "RETURN n AS record ORDER BY n.opaque_key LIMIT $family_limit"
)
_RELATION_TRAVERSAL = (
    "MATCH (membership:AutomaticMembership {generation_key:$generation_key}) "
    "WHERE membership.automatic_membership_key IN split($seed_keys_csv, ',') "
    "OR membership.entity_key IN split($seed_keys_csv, ',') "
    "MATCH (seed:ProjectedEntity {generation_key:$generation_key})"
    "-[:ENTITY_MEMBERSHIP]->(membership) "
    "MATCH p=(seed)"
    "-[:PROJECTED_RELATION*1..2]-(target:ProjectedEntity "
    "{generation_key:$generation_key}) "
    "WHERE seed.opaque_key IN split($seed_keys_csv, ',') "
    "AND length(p) <= $max_depth "
    "MATCH (r:ProjectedRelation {generation_key:$generation_key}) "
    "WHERE any(edge IN relationships(p) WHERE edge.relation_key = r.relation_key) "
    "RETURN r AS record ORDER BY r.opaque_key LIMIT $family_limit"
)
_EVIDENCE_TRAVERSAL = (
    "MATCH (r:ProjectedRelation {generation_key:$generation_key})"
    "-[edge:RELATION_EVIDENCE]->(c:ProjectedChunk "
    "{generation_key:$generation_key}) "
    "WHERE c.document_key IN split($authorized_document_keys_csv, ',') "
    "MATCH (n:ProjectedEvidence {generation_key:$generation_key}) "
    "WHERE n.opaque_key = edge.evidence_key "
    "RETURN n AS record ORDER BY n.opaque_key LIMIT $family_limit"
)
_MENTION_TRAVERSAL = (
    "MATCH (entity:ProjectedEntity {generation_key:$generation_key})"
    "-[edge:ENTITY_MENTION]->(c:ProjectedChunk "
    "{generation_key:$generation_key}) "
    "WHERE c.document_key IN split($authorized_document_keys_csv, ',') "
    "MATCH (n:ProjectedEntityMention {generation_key:$generation_key}) "
    "WHERE n.opaque_key = edge.mention_key "
    "RETURN n AS record ORDER BY n.opaque_key LIMIT $family_limit"
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
    if len(maxima) != len(FAMILIES) or any(
        type(value) is not int or not 1 <= value <= 5_000 for value in maxima
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
        if topology_parameters is not None and label == "ProjectedEntity":
            query = _ENTITY_TRAVERSAL
        elif topology_parameters is not None and label == "ProjectedRelation":
            query = _RELATION_TRAVERSAL
        elif topology_parameters is not None and label == "ProjectedEvidence":
            query = _EVIDENCE_TRAVERSAL
        elif topology_parameters is not None and label == "ProjectedEntityMention":
            query = _MENTION_TRAVERSAL
        else:
            query = (
                f"MATCH (n:{label} {{generation_key:$generation_key}}) "
                "RETURN n AS record ORDER BY n.opaque_key"
            )
        family_parameters = dict(parameters)
        if topology_parameters is not None:
            family_parameters["family_limit"] = (
                maximum + 1 if reject_full_pages else maximum
            )
        rows = driver.execute_read(
            query,
            family_parameters,
            timeout_seconds=timeout,
            max_records=maximum + 1 if reject_full_pages else maximum,
        )
        if reject_full_pages and len(rows) > maximum:
            raise ValueError("bounded Memgraph family read was truncated")
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
