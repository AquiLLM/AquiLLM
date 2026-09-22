# ruff: noqa: E501,E701,E702
"""Scoped database queries for direct seed lookup."""

from __future__ import annotations

from apps.knowledge_graph.retrieval.direct_seed_contracts import DirectResolutionTier

from .direct_seed_types import DirectSeedCandidateRow, DirectSeedScopeV1


# fmt: off
def _load_membership_states(**options: object) -> tuple[dict[str, object], ...]:
    from apps.knowledge_graph.models import CollectionGraphMembershipState
    fields = ("collection_id", "active_artifact_id", "registry_epoch", "membership_checksum", "resolver_version", "resolution_config_checksum")
    return tuple(CollectionGraphMembershipState.objects.using(options["using"]).filter(collection_id__in=options["collection_ids"]).order_by("collection_id").values(*fields))


def _load_candidate_rows(**options: object) -> tuple[DirectSeedCandidateRow, ...]:
    from django.db.models import F, FloatField, OuterRef, Q, Subquery, Value
    from django.db.models.expressions import ExpressionWrapper
    from pgvector.django import CosineDistance

    from apps.knowledge_graph.models import CanonicalEntityLink, CollectionEntity

    scope = options["scope"]
    assert type(scope) is DirectSeedScopeV1
    membership_scope = Q()
    for state in options["membership_states"]:
        membership_scope |= Q(collection_id=state["collection_id"], artifact_id=state["active_artifact_id"], collection__graph_membership_state__active_artifact_id=state["active_artifact_id"], collection__graph_membership_state__registry_epoch=state["registry_epoch"], collection__graph_membership_state__membership_checksum=state["membership_checksum"], collection__graph_membership_state__resolver_version=state["resolver_version"], collection__graph_membership_state__resolution_config_checksum=state["resolution_config_checksum"])
    assert options["automatic_only"] is True
    automatic = CanonicalEntityLink.objects.using(options["using"]).filter(
        collection_entity_id=OuterRef("pk"), status="active", outcome="automatic",
        resolver_version=scope.resolver_version, canonical_entity__status="active",
        canonical_entity__resolver_version=scope.resolver_version, canonical_entity__entity_type=OuterRef("entity_type"),
        canonical_entity__version_signature=OuterRef("version_signature"),
    ).values("canonical_entity_id")[:1]
    tier = options["tier"]
    alias_filters = {
        "document_links__document_entity__mention_links__status": "active",
        # Mention assignments belong to the document resolver, which can evolve
        # independently of the selected collection's assembly resolver.
        "document_links__document_entity__mention_links__resolver_version": F(
            "document_links__document_entity__artifact__resolver_version"
        ),
        "document_links__document_entity__mention_links__mention__artifact_id__in": scope.selected_document_artifact_ids,
        "document_links__document_entity__mention_links__mention__document_id__in": scope.selected_document_ids,
        "document_links__document_entity__mention_links__mention__artifact__status__in": ("active", "superseded"),
        "document_links__document_entity__mention_links__mention__artifact__evaluation_only": False,
        "document_links__document_entity__mention_links__mention__artifact__ontology_checksum": scope.ontology_checksum,
        "document_links__document_entity__mention_links__mention__entity_type": options["ontology_type"],
        str(options["lookup_field"]): options["lookup"],
    } if tier is DirectResolutionTier.ALIAS else {}
    query = (
        CollectionEntity.objects.using(options["using"])
        .filter(
            membership_scope,
            artifact_id__in=scope.selected_artifact_ids, collection_id__in=scope.selected_collection_ids,
            artifact__status="active", artifact__evaluation_only=False,
            artifact__ontology_checksum=scope.ontology_checksum, status="active", entity_type=options["ontology_type"],
            document_links__artifact_id__in=scope.selected_artifact_ids, document_links__status="active",
            document_links__outcome="automatic", document_links__resolver_version=scope.resolver_version,
            document_links__artifact__status="active", document_links__artifact__evaluation_only=False, document_links__artifact__ontology_checksum=scope.ontology_checksum,
            document_links__manifest_input__artifact_id=F("artifact_id"), document_links__manifest_input__collection_id=F("collection_id"),
            document_links__manifest_input__document_id__in=scope.selected_document_ids,
            document_links__manifest_input__document_artifact_id__in=scope.selected_document_artifact_ids,
            document_links__document_entity__artifact_id__in=scope.selected_document_artifact_ids,
            document_links__document_entity__document_id__in=scope.selected_document_ids,
            document_links__manifest_input__document_artifact_id=F("document_links__document_entity__artifact_id"), document_links__manifest_input__document_id=F("document_links__document_entity__document_id"),
            document_links__document_entity__status="active", document_links__document_entity__artifact__status__in=("active", "superseded"), document_links__document_entity__artifact__evaluation_only=False,
            document_links__document_entity__artifact__ontology_checksum=scope.ontology_checksum,
            **alias_filters,
        )
        .annotate(automatic_canonical_entity_id=Subquery(automatic))
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
    rows = tuple(query.distinct().order_by("-similarity", "pk").values("id", "artifact_id", "entity_type", "automatic_canonical_entity_id", "similarity")[: int(options["limit"]) + 1])
    if len(rows) > int(options["limit"]):
        raise ValueError("candidate result exceeds its hard cap")
    return tuple(DirectSeedCandidateRow(int(row["id"]), int(row["artifact_id"]), str(row["entity_type"]), row["automatic_canonical_entity_id"], float(row["similarity"])) for row in rows)
