from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import F, Q

from apps.documents.models import TextChunk

from .artifacts import GraphArtifact
from .entities import CollectionEntity, EntityMention, ResolutionStatus


class RelationMention(models.Model):
    """Extracted relation evidence whose endpoint spans remain first-class mentions."""

    artifact = models.ForeignKey(
        GraphArtifact,
        on_delete=models.CASCADE,
        related_name="relation_mentions",
    )
    document_id = models.UUIDField()
    chunk = models.ForeignKey(
        TextChunk,
        on_delete=models.CASCADE,
        related_name="graph_relation_mentions",
    )
    head = models.ForeignKey(
        EntityMention,
        on_delete=models.CASCADE,
        related_name="outgoing_relation_mentions",
    )
    tail = models.ForeignKey(
        EntityMention,
        on_delete=models.CASCADE,
        related_name="incoming_relation_mentions",
    )
    relation_type = models.CharField(max_length=128)
    extraction_confidence = models.FloatField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(extraction_confidence__gte=0)
                & Q(extraction_confidence__lte=1),
                name="kg_relation_mention_conf_range",
            ),
            models.CheckConstraint(
                condition=~Q(head=F("tail")),
                name="kg_relation_mention_endpoints",
            ),
        ]
        indexes = [
            models.Index(
                fields=["artifact", "document_id", "chunk"],
                name="kg_rel_mention_evidence_idx",
            ),
            models.Index(
                fields=["artifact", "relation_type"],
                name="kg_rel_mention_type_idx",
            ),
        ]

    def clean(self):
        super().clean()
        errors: dict[str, str] = {}
        if self.chunk_id and self.document_id != self.chunk.doc_id:
            errors["document_id"] = "Relation document must match its chunk."
        if self.head_id and self.tail_id:
            for endpoint_name, endpoint in (("head", self.head), ("tail", self.tail)):
                if endpoint.artifact_id != self.artifact_id:
                    errors[endpoint_name] = "Relation endpoint artifact must match."
                elif endpoint.document_id != self.document_id:
                    errors[endpoint_name] = "Relation endpoint document must match."
                elif endpoint.chunk_id != self.chunk_id:
                    errors[endpoint_name] = "Relation endpoint chunk must match."
        if errors:
            raise ValidationError(errors)


class CollectionRelation(models.Model):
    """Aggregated, status-preserving edge inside one collection artifact."""

    Status = ResolutionStatus

    artifact = models.ForeignKey(
        GraphArtifact,
        on_delete=models.CASCADE,
        related_name="collection_relations",
    )
    source = models.ForeignKey(
        CollectionEntity,
        on_delete=models.CASCADE,
        related_name="outgoing_relations",
    )
    relation_type = models.CharField(max_length=128)
    target = models.ForeignKey(
        CollectionEntity,
        on_delete=models.CASCADE,
        related_name="incoming_relations",
    )
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    support_count = models.PositiveIntegerField(default=0)
    confidence = models.FloatField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["artifact", "source", "relation_type", "target"],
                name="kg_collection_relation_unique",
            ),
            models.CheckConstraint(
                condition=Q(support_count__gte=0),
                name="kg_collection_relation_support",
            ),
            models.CheckConstraint(
                condition=Q(confidence__gte=0) & Q(confidence__lte=1),
                name="kg_collection_relation_conf",
            ),
            models.CheckConstraint(
                condition=~Q(source=F("target")),
                name="kg_collection_relation_endpoints",
            ),
        ]
        indexes = [
            models.Index(
                fields=["artifact", "source", "target", "relation_type"],
                name="kg_collection_relation_idx",
            ),
        ]

    def clean(self):
        super().clean()
        errors: dict[str, str] = {}
        if self.source_id and self.source.artifact_id != self.artifact_id:
            errors["source"] = "Source artifact must match relation artifact."
        if self.target_id and self.target.artifact_id != self.artifact_id:
            errors["target"] = "Target artifact must match relation artifact."
        if self.source_id and self.target_id:
            if self.source.collection_id != self.target.collection_id:
                errors["target"] = (
                    "Relation endpoints must belong to the same collection."
                )
        if errors:
            raise ValidationError(errors)


class CollectionRelationEvidence(models.Model):
    """One retained supporting extraction for a collection relation."""

    relation = models.ForeignKey(
        CollectionRelation,
        on_delete=models.CASCADE,
        related_name="evidence",
    )
    relation_mention = models.ForeignKey(
        RelationMention,
        on_delete=models.CASCADE,
        related_name="collection_evidence_links",
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["relation", "relation_mention"],
                name="kg_relation_evidence_unique",
            )
        ]

    def clean(self):
        super().clean()
        if (
            self.relation_id
            and self.relation_mention_id
            and self.relation.relation_type != self.relation_mention.relation_type
        ):
            raise ValidationError(
                {"relation_mention": "Evidence relation type must match the relation."}
            )
