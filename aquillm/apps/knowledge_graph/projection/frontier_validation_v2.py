"""Page-level generation, membership, semantic and source-authority closure."""

from .frontier_reads_v2 import schema_error


def validate_relation_entry(page, entry):
    if (
        len(page.relations) != 1
        or len(page.relation_semantics) != 1
        or len(entry["order"]) != 5
    ):
        raise schema_error()
    relation, semantic = page.relations[0], page.relation_semantics[0]
    _, relation_key, origin, target, direction = entry["order"]
    forward = (origin, target) == (
        relation.source_entity_key,
        relation.target_entity_key,
    )
    reverse = (origin, target) == (
        relation.target_entity_key,
        relation.source_entity_key,
    )
    expected = (
        "undirected"
        if semantic.direction == "undirected"
        else ("forward" if forward else "reverse_directed")
    )
    if (
        relation_key != relation.relation_key
        or not (forward or reverse)
        or direction != expected
    ):
        raise schema_error()


def validate_source_page(page, *, selected, ready):
    entities = {r.entity_key: r for r in page.nodes}
    members = {r.entity_key: r for r in page.memberships}
    if (
        len(entities) != len(page.nodes)
        or len(members) != len(page.memberships)
        or set(entities) != set(members)
    ):
        raise schema_error()
    for row in page.nodes:
        if (row.generation_key, row.collection_key, row.artifact_key) != (
            selected.generation_key,
            selected.collection_key,
            selected.active_artifact_key,
        ):
            raise schema_error()
        member = members[row.entity_key]
        if (
            member.decision_checksum,
            member.resolver_version,
            member.resolution_config_checksum,
        ) != (
            selected.membership_checksum,
            selected.resolver_version,
            selected.resolution_config_checksum,
        ):
            raise schema_error()
    authorized = {
        r.document_key
        for r in ready.authorized_documents
        if r.generation_key == selected.generation_key
    }
    documents = {r.document_key: r for r in page.documents}
    chunks = {r.chunk_key: r for r in page.chunks}
    provenance = {r.scope_key: r for r in page.provenance}
    shared = (
        "ontology_version",
        "ontology_checksum",
        "extractor_version",
        "orchestration_version",
    )
    if any(len({getattr(r, name) for r in page.provenance}) > 1 for name in shared):
        raise schema_error()
    for row in page.documents:
        if row.generation_key != selected.generation_key:
            raise schema_error()
    for row in page.provenance:
        if (
            row.collection_key != selected.collection_key
            or row.ontology_checksum != selected.ontology_checksum
        ):
            raise schema_error()
        if row.scope_type == "collection" and (
            row.artifact_key != selected.active_artifact_key
            or row.scope_key != selected.collection_key
            or row.resolver_version != selected.resolver_version
            or row.resolution_config_checksum != selected.resolution_config_checksum
            or row.embedding_model_signature != selected.embedding_model_signature
        ):
            raise schema_error()
    relations = {r.relation_key: r for r in page.relations}
    semantics = {r.relation_type: r for r in page.relation_semantics}
    for row in page.relations:
        semantic = semantics.get(row.relation_type)
        if (
            row.artifact_key != selected.active_artifact_key
            or row.source_entity_key not in entities
            or row.target_entity_key not in entities
            or row.source_entity_key == row.target_entity_key
            or semantic is None
            or semantic.artifact_key != row.artifact_key
        ):
            raise schema_error()
    for row in (*page.relation_evidence, *page.mention_evidence):
        chunk, document, source = (
            chunks.get(row.chunk_key),
            documents.get(row.document_key),
            provenance.get(row.document_key),
        )
        if (
            chunk is None
            or document is None
            or source is None
            or source.scope_type != "document"
            or (chunk.document_key, chunk.chunk_number)
            != (row.document_key, row.chunk_number)
        ):
            raise schema_error()
        if hasattr(row, "relation_key"):
            relation = relations.get(row.relation_key)
            collection = provenance.get(selected.collection_key)
            if (
                relation is None
                or collection is None
                or row.source_document_key != row.document_key
                or row.artifact_key != source.artifact_key
                or row.relation_type != relation.relation_type
                or row.ontology_checksum != selected.ontology_checksum
                or row.assembly_config_checksum != collection.assembly_config_checksum
            ):
                raise schema_error()
        elif row.entity_key not in entities:
            raise schema_error()
    # False is an inaccessible observation, never a transit edge. Corruption
    # above is still checked before this authorization disposition.
    return all(
        r.document_key in authorized
        for r in (*page.relation_evidence, *page.mention_evidence)
    )
