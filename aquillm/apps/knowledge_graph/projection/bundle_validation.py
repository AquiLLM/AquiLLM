from __future__ import annotations


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


def validate_bundle(bundle: object) -> None:
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
        "ProjectedPhysicalRelationV1:relation_key|relation_semantics:"
        "ProjectedRelationSemanticsV1:artifact_key relation_type "
        "semantics_key|evidence:"
        "ProjectedRelationEvidenceV1:evidence_key|artifact_provenance:"
        "ProjectedArtifactProvenanceV1:scope_type scope_key artifact_key|"
        "entity_mentions:"
        "ProjectedEntityMentionEvidenceV1:entity_key provenance_key mention_key"
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
    semantics = {row.relation_type: row for row in bundle.relation_semantics}
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
    if (
        len(semantics) != len(bundle.relation_semantics)
        or set(semantics) != {row.relation_type for row in bundle.relations}
        or any(row.artifact_key != marker.artifact_key for row in semantics.values())
    ):
        raise ValueError("relation semantics closure is broken")
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
        "ontology_version ontology_checksum extractor_version "
        "orchestration_version"
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
    if any(
        row.entity_key not in entities
        or row.chunk_key not in chunks
        or chunks.get(row.chunk_key) != (row.document_key, row.chunk_number)
        or row.document_key not in documents_by_scope
        for row in bundle.entity_mentions
    ):
        raise ValueError("entity mention evidence closure is broken")
    pairs = (
        "entity_count:entities automatic_membership_count:automatic_memberships "
        "document_count:documents chunk_count:chunks "
        "relation_semantics_count:relation_semantics relation_count:relations "
        "evidence_count:evidence entity_mention_count:entity_mentions "
        "artifact_provenance_count:artifact_provenance"
    ).split()
    actual = tuple(getattr(bundle.counts, pair.split(":")[0]) for pair in pairs)
    expected = tuple(len(getattr(bundle, pair.split(":")[1])) for pair in pairs)
    if actual != expected:
        raise ValueError("projection count agreement is broken")
