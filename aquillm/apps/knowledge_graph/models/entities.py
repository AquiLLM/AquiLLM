from __future__ import annotations

from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import F, Q
from pgvector.django import VectorField

from apps.documents.models import TextChunk

from .artifacts import GraphArtifact


class ResolutionStatus(models.TextChoices):
    ACTIVE = "active", "Active"
    SUPPRESSED = "suppressed", "Suppressed"
    REJECTED = "rejected", "Rejected"
    SUPERSEDED = "superseded", "Superseded"


class EntityMention(models.Model):
    """Exact entity evidence extracted from one persisted chunk."""

    class PositionBasis(models.TextChoices):
        DOCUMENT_GLOBAL = "document_global", "Document global"
        CHUNK_CONTENT = "chunk_content", "Chunk content"

    _IMAGE_CONTENT_OBJECT_TYPES = frozenset(
        {
            ("apps_documents", "documentfigure"),
            ("apps_documents", "handwrittennotesdocument"),
            ("apps_documents", "imageuploaddocument"),
        }
    )

    artifact = models.ForeignKey(
        GraphArtifact,
        on_delete=models.CASCADE,
        related_name="entity_mentions",
    )
    document_id = models.UUIDField()
    chunk = models.ForeignKey(
        TextChunk,
        on_delete=models.CASCADE,
        related_name="graph_entity_mentions",
    )
    start = models.IntegerField()
    end = models.IntegerField()
    position_basis = models.CharField(max_length=24, choices=PositionBasis.choices)
    raw_text = models.TextField()
    normalized_text = models.TextField()
    entity_type = models.CharField(max_length=128)
    extraction_confidence = models.FloatField()
    content_object_type = models.ForeignKey(
        ContentType,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="+",
    )
    content_object_id = models.UUIDField(null=True, blank=True)
    content_object = GenericForeignKey("content_object_type", "content_object_id")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(start__gte=0) & Q(end__gt=F("start")),
                name="kg_mention_nonempty_span",
            ),
            models.CheckConstraint(
                condition=Q(extraction_confidence__gte=0)
                & Q(extraction_confidence__lte=1),
                name="kg_mention_confidence_range",
            ),
            models.CheckConstraint(
                condition=(
                    Q(content_object_type__isnull=True, content_object_id__isnull=True)
                    | Q(
                        content_object_type__isnull=False,
                        content_object_id__isnull=False,
                    )
                ),
                name="kg_mention_content_object_pair",
            ),
        ]
        indexes = [
            models.Index(
                fields=["document_id", "chunk"], name="kg_mention_doc_chunk_idx"
            ),
            models.Index(
                fields=["normalized_text", "entity_type"],
                name="kg_mention_norm_type_idx",
            ),
        ]

    def clean(self):
        super().clean()
        errors: dict[str, str] = {}
        if self.start < 0 or self.end <= self.start:
            errors["end"] = "Mention spans must be nonnegative and nonempty."
        if not 0 <= self.extraction_confidence <= 1:
            errors["extraction_confidence"] = "Confidence must be in [0, 1]."
        chunk = self.chunk
        if self.document_id and chunk.doc_id and self.document_id != chunk.doc_id:
            errors["document_id"] = "Mention document_id must match its chunk."
        if chunk.modality == TextChunk.Modality.TEXT:
            if self.position_basis != self.PositionBasis.DOCUMENT_GLOBAL:
                errors["position_basis"] = (
                    "Text evidence requires document_global positions."
                )
            if self.content_object_type_id or self.content_object_id:
                errors["content_object_id"] = (
                    "Text evidence must not include image provenance."
                )
        elif chunk.modality == TextChunk.Modality.IMAGE:
            if self.position_basis != self.PositionBasis.CHUNK_CONTENT:
                errors["position_basis"] = (
                    "Image evidence requires chunk_content positions."
                )
            if not self.content_object_type_id or not self.content_object_id:
                errors["content_object_id"] = (
                    "Image evidence requires a typed content object."
                )
            elif (
                self.content_object_type.app_label,
                self.content_object_type.model,
            ) not in self._IMAGE_CONTENT_OBJECT_TYPES:
                errors["content_object_type"] = (
                    "Image content object type must identify an image document."
                )
            elif chunk.doc_id != self.content_object_id:
                errors["content_object_id"] = (
                    "Image content object must match the chunk document."
                )
        if errors:
            raise ValidationError(errors)


class DocumentEntity(models.Model):
    """Entity resolved within a document artifact."""

    Status = ResolutionStatus

    artifact = models.ForeignKey(
        GraphArtifact,
        on_delete=models.CASCADE,
        related_name="document_entities",
    )
    document_id = models.UUIDField()
    label = models.TextField()
    normalized_label = models.TextField()
    entity_type = models.CharField(max_length=128)
    identifier = models.CharField(max_length=255, blank=True, default="")
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["artifact", "document_id", "entity_type", "normalized_label"],
                name="kg_document_entity_unique",
            )
        ]
        indexes = [
            models.Index(
                fields=["document_id", "entity_type", "normalized_label"],
                name="kg_doc_entity_lookup_idx",
            )
        ]

    def clean(self):
        super().clean()
        if (
            self.artifact_id
            and self.artifact.scope_type != GraphArtifact.ScopeType.DOCUMENT
        ):
            raise ValidationError(
                {"artifact": "Document entities require a document artifact."}
            )
        if self.artifact_id and self.document_id != self.artifact.scope_id:
            raise ValidationError(
                {"document_id": "Document entity must match artifact scope."}
            )


class DocumentEntityMention(models.Model):
    """Auditable assignment of one mention to one document entity."""

    Status = ResolutionStatus

    document_entity = models.ForeignKey(
        DocumentEntity,
        on_delete=models.CASCADE,
        related_name="mention_links",
    )
    mention = models.OneToOneField(
        EntityMention,
        on_delete=models.CASCADE,
        related_name="document_entity_link",
    )
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    reason = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        indexes = [
            models.Index(
                fields=["status", "document_entity"],
                name="kg_doc_mention_status_idx",
            )
        ]

    def clean(self):
        super().clean()
        if self.document_entity_id and self.mention_id:
            if self.document_entity.artifact_id != self.mention.artifact_id:
                raise ValidationError(
                    {"mention": "Mention and document entity artifacts must match."}
                )
            if self.document_entity.document_id != self.mention.document_id:
                raise ValidationError(
                    {"mention": "Mention and document entity documents must match."}
                )


class CollectionEntity(models.Model):
    """Collection-scoped resolved entity with optional retrieval embedding."""

    Status = ResolutionStatus

    artifact = models.ForeignKey(
        GraphArtifact,
        on_delete=models.CASCADE,
        related_name="collection_entities",
    )
    collection_id = models.UUIDField()
    label = models.TextField()
    normalized_label = models.TextField()
    entity_type = models.CharField(max_length=128)
    identifier = models.CharField(max_length=255, blank=True, default="")
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    embedding = VectorField(dimensions=1024, blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["artifact", "entity_type", "normalized_label", "identifier"],
                name="kg_collection_entity_unique",
            )
        ]
        indexes = [
            models.Index(
                fields=["collection_id", "entity_type", "normalized_label"],
                name="kg_collection_entity_lookup",
            )
        ]

    def clean(self):
        super().clean()
        if (
            self.artifact_id
            and self.artifact.scope_type != GraphArtifact.ScopeType.COLLECTION
        ):
            raise ValidationError(
                {"artifact": "Collection entities require a collection artifact."}
            )
        if self.artifact_id and self.collection_id != self.artifact.scope_id:
            raise ValidationError(
                {"collection_id": "Collection entity must match artifact scope."}
            )


class CanonicalEntity(models.Model):
    """Internal cross-collection identity without evidence or access grants."""

    Status = ResolutionStatus

    label = models.TextField()
    normalized_label = models.TextField()
    entity_type = models.CharField(max_length=128)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    embedding = VectorField(dimensions=1024, blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        indexes = [
            models.Index(
                fields=["entity_type", "normalized_label"],
                name="kg_canonical_entity_lookup",
            )
        ]
