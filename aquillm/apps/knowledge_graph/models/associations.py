from django.db import models
from django.db.models import Q

from .entities import (
    CanonicalEntity,
    CollectionEntity,
    DocumentEntity,
    ResolutionStatus,
)


class CollectionEntityDocumentLink(models.Model):
    """Versioned resolver decision linking document and collection nodes."""

    Status = ResolutionStatus

    document_entity = models.ForeignKey(
        DocumentEntity,
        on_delete=models.CASCADE,
        related_name="collection_links",
    )
    collection_entity = models.ForeignKey(
        CollectionEntity,
        on_delete=models.CASCADE,
        related_name="document_links",
    )
    score = models.FloatField()
    method = models.CharField(max_length=64)
    resolver_version = models.CharField(max_length=128)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    reason = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(score__gte=0) & Q(score__lte=1),
                name="kg_doc_collection_score_range",
            ),
            models.UniqueConstraint(
                fields=["document_entity", "collection_entity", "resolver_version"],
                name="kg_doc_collection_entity_link_unique",
            ),
        ]
        indexes = [
            models.Index(
                fields=["status", "collection_entity"],
                name="kg_doc_link_collection_idx",
            ),
            models.Index(
                fields=["status", "document_entity"],
                name="kg_doc_link_document_idx",
            ),
        ]

class CanonicalEntityLink(models.Model):
    """Explicit resolver decision from a collection node to internal identity."""

    Status = ResolutionStatus

    collection_entity = models.ForeignKey(
        CollectionEntity,
        on_delete=models.CASCADE,
        related_name="canonical_links",
    )
    canonical_entity = models.ForeignKey(
        CanonicalEntity,
        on_delete=models.CASCADE,
        related_name="collection_links",
    )
    score = models.FloatField()
    method = models.CharField(max_length=64)
    resolver_version = models.CharField(max_length=128)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    reason = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(score__gte=0) & Q(score__lte=1),
                name="kg_collection_canonical_score_range",
            ),
            models.UniqueConstraint(
                fields=["collection_entity", "canonical_entity", "resolver_version"],
                name="kg_collection_canonical_link_unique",
            ),
        ]
        indexes = [
            models.Index(
                fields=["status", "canonical_entity"],
                name="kg_can_link_canonical_idx",
            ),
            models.Index(
                fields=["status", "collection_entity"],
                name="kg_can_link_collection_idx",
            ),
        ]
