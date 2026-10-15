# ruff: noqa: E501,E701,E702
"""Scoped database queries for direct seed lookup."""

from __future__ import annotations

from apps.knowledge_graph.retrieval.direct_seed_contracts import DirectResolutionTier

from .direct_seed_types import DirectSeedScopeV1


# fmt: off
def _candidate_queryset(**options: object):
    from django.db.models import (
        Count,
        Exists,
        F,
        FloatField,
        OuterRef,
        Q,
        Subquery,
        Value,
    )
    from django.db.models.expressions import ExpressionWrapper
    from pgvector.django import CosineDistance

    from apps.knowledge_graph.models import (
        CanonicalEntityLink,
        CollectionEntity,
        CollectionEntityDocumentLink,
        CollectionGraphMembershipState,
    )

    scope = options["scope"]
    assert type(scope) is DirectSeedScopeV1
    tier = options["tier"]
    membership_scope = Q()
    if tier is DirectResolutionTier.NAME:
        # Paired reads favored this state shape for names; alias retained A's join shape.
        for state in options["membership_states"]:
            membership_scope |= Q(collection_id=state["collection_id"], active_artifact_id=state["active_artifact_id"], registry_epoch=state["registry_epoch"], membership_checksum=state["membership_checksum"], resolver_version=state["resolver_version"], resolution_config_checksum=state["resolution_config_checksum"])
        membership = CollectionGraphMembershipState.objects.using(options["using"]).filter(
            membership_scope,
            collection_id=OuterRef("collection_id"),
            active_artifact_id=OuterRef("artifact_id"),
        )
        membership_predicate = Exists(membership)
    else:
        for state in options["membership_states"]:
            membership_scope |= Q(collection_id=state["collection_id"], artifact_id=state["active_artifact_id"], collection__graph_membership_state__active_artifact_id=state["active_artifact_id"], collection__graph_membership_state__registry_epoch=state["registry_epoch"], collection__graph_membership_state__membership_checksum=state["membership_checksum"], collection__graph_membership_state__resolver_version=state["resolver_version"], collection__graph_membership_state__resolution_config_checksum=state["resolution_config_checksum"])
        membership_predicate = membership_scope
    assert options["automatic_only"] is True
    active_count = CanonicalEntityLink.objects.using(options["using"]).filter(
        collection_entity_id=OuterRef("pk"), status="active", outcome="automatic",
        canonical_entity__status="active",
    ).order_by().values("collection_entity_id").annotate(total=Count("pk")).values("total")[:1]
    automatic = CanonicalEntityLink.objects.using(options["using"]).filter(
        collection_entity_id=OuterRef("pk"), status="active", outcome="automatic",
        canonical_entity__status="active",
        canonical_entity__resolver_version=F("resolver_version"), canonical_entity__entity_type=OuterRef("entity_type"),
        canonical_entity__version_signature=OuterRef("version_signature"),
    ).values("canonical_entity_id")[:1]
    alias_filters = {
        "document_entity__mention_links__status": "active",
        # Mention assignments belong to the document resolver, which can evolve
        # independently of the selected collection's assembly resolver.
        "document_entity__mention_links__resolver_version": F(
            "document_entity__artifact__resolver_version"
        ),
        "document_entity__mention_links__mention__artifact_id__in": scope.selected_document_artifact_ids,
        "document_entity__mention_links__mention__document_id__in": scope.selected_document_ids,
        "document_entity__mention_links__mention__artifact__status__in": ("active", "superseded"),
        "document_entity__mention_links__mention__artifact__evaluation_only": False,
        "document_entity__mention_links__mention__artifact__ontology_checksum": scope.ontology_checksum,
        "document_entity__mention_links__mention__entity_type": options["ontology_type"],
        str(options["lookup_field"]).removeprefix("document_links__"): options["lookup"],
    } if tier is DirectResolutionTier.ALIAS else {}
    witness = CollectionEntityDocumentLink.objects.using(options["using"]).filter(
        collection_entity_id=OuterRef("pk"),
        artifact_id__in=scope.selected_artifact_ids, status="active",
        outcome="automatic", resolver_version=scope.resolver_version,
        artifact__status="active", artifact__evaluation_only=False, artifact__ontology_checksum=scope.ontology_checksum,
        manifest_input__artifact_id=OuterRef("artifact_id"), manifest_input__collection_id=OuterRef("collection_id"),
        manifest_input__document_id__in=scope.selected_document_ids,
        manifest_input__document_artifact_id__in=scope.selected_document_artifact_ids,
        document_entity__artifact_id__in=scope.selected_document_artifact_ids,
        document_entity__document_id__in=scope.selected_document_ids,
        manifest_input__document_artifact_id=F("document_entity__artifact_id"), manifest_input__document_id=F("document_entity__document_id"),
        document_entity__status="active", document_entity__artifact__status__in=("active", "superseded"), document_entity__artifact__evaluation_only=False,
        document_entity__artifact__ontology_checksum=scope.ontology_checksum,
        **alias_filters,
    )
    query = (
        CollectionEntity.objects.using(options["using"])
        .filter(
            membership_predicate,
            Exists(witness),
            artifact_id__in=scope.selected_artifact_ids, collection_id__in=scope.selected_collection_ids,
            artifact__status="active", artifact__evaluation_only=False,
            artifact__ontology_checksum=scope.ontology_checksum, status="active", entity_type=options["ontology_type"],
        )
        .annotate(automatic_canonical_entity_id=Subquery(automatic), automatic_link_count=Subquery(active_count))
    )
    if tier is DirectResolutionTier.EMBEDDING:
        similarity = ExpressionWrapper(
            Value(1.0) - CosineDistance("embedding", options["embedding"]),
            output_field=FloatField(),
        )
        query = query.filter(embedding__isnull=False, embedding_model_signature=options["model_signature"]).annotate(similarity=similarity).filter(similarity__gt=0.0, similarity__gte=options["minimum_similarity"])
    else:
        if tier is not DirectResolutionTier.ALIAS:
            query = query.filter(**{str(options["lookup_field"]): options["lookup"]})
        query = query.annotate(similarity=Value(1.0, output_field=FloatField()))
    return query.distinct().order_by("-similarity", "pk").values("id", "artifact_id", "entity_type", "automatic_canonical_entity_id", "automatic_link_count", "similarity")[: int(options["limit"]) + 1]

_NATIVE_CANDIDATE_QUERYSET = _candidate_queryset
