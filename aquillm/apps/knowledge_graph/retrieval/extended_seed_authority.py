"""The exact fresh projection-state predicate shared by seed repositories."""


def _authority_query(*, authority, using):
    from apps.knowledge_graph.models import CollectionGraphProjection

    return CollectionGraphProjection.objects.using(using).filter(
        pk=authority.projection_id,
        generation_key=authority.generation_id,
        state="ready",
        collection_id=authority.collection_id,
        collection_pk_snapshot=authority.collection_id,
        artifact_id=authority.artifact_id,
        artifact_pk_snapshot=authority.artifact_id,
        graph_checksum=authority.graph_checksum,
        private_mapping_checksum=authority.private_mapping_checksum,
        schema_version=authority.schema_version,
        projection_version=authority.projection_version,
        identifier_key_version=authority.identifier_key_version,
        membership_epoch=authority.membership_epoch,
        membership_checksum=authority.membership_checksum,
        collection__graph_membership_state__active_artifact_id=authority.artifact_id,
        collection__graph_membership_state__registry_epoch=authority.membership_epoch,
        collection__graph_membership_state__membership_checksum=authority.membership_checksum,
        collection__graph_membership_state__resolver_version=authority.resolver_version,
        collection__graph_membership_state__resolution_config_checksum=authority.resolution_config_checksum,
        artifact__status="active",
        artifact__evaluation_only=False,
        artifact__scope_type="collection",
        artifact__scope_id=str(authority.collection_id),
        artifact__collection_scope_id=authority.collection_id,
        artifact__ontology_version=authority.ontology_version,
        artifact__ontology_checksum=authority.ontology_checksum,
        artifact__resolver_version=authority.resolver_version,
        artifact__resolution_config_checksum=authority.resolution_config_checksum,
        artifact__embedding_model_signature=authority.embedding_model_signature,
    )
