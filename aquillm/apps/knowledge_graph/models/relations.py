from math import isfinite
from unicodedata import normalize as unicode_normalize

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import F, Q

from apps.documents.models import TextChunk

from .artifacts import GraphArtifact, ImmutableGraphQuerySet, ValidatedGraphModel
from .associations import CollectionEntityDocumentLink
from .entities import (
    CollectionEntity,
    DocumentEntityMention,
    EntityMention,
    ResolutionStatus,
)


class RelationMention(ValidatedGraphModel):
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

    _IMMUTABLE_FIELDS = (
        "artifact",
        "artifact_id",
        "document_id",
        "chunk",
        "chunk_id",
        "head",
        "head_id",
        "tail",
        "tail_id",
        "relation_type",
        "extraction_confidence",
        "metadata",
        "created_at",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS

    objects = models.Manager.from_queryset(ImmutableGraphQuerySet)()

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

    def _raw_validation_errors(self) -> dict[str, str]:
        value = self.extraction_confidence
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            return {
                "extraction_confidence": (
                    "Confidence must be a finite non-boolean number."
                )
            }
        return {}

    def _endpoint_has_relation_chunk_observation(self, endpoint: EntityMention) -> bool:
        if endpoint.chunk_id == self.chunk_id:
            return True
        if (
            endpoint.position_basis != EntityMention.PositionBasis.DOCUMENT_GLOBAL
            or self.chunk.modality != TextChunk.Modality.TEXT
        ):
            return False
        observations = endpoint.metadata.get("observations", [])
        if not isinstance(observations, list):
            return False
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            local_start = observation.get("local_start")
            local_end = observation.get("local_end")
            if (
                observation.get("chunk_id") != self.chunk_id
                or observation.get("modality") != TextChunk.Modality.TEXT
                or observation.get("position_basis")
                != EntityMention.PositionBasis.DOCUMENT_GLOBAL
                or observation.get("start") != endpoint.start
                or observation.get("end") != endpoint.end
                or type(local_start) is not int
                or type(local_end) is not int
                or not 0 <= local_start < local_end <= len(self.chunk.content)
                or self.chunk.start_position + local_start != endpoint.start
                or self.chunk.start_position + local_end != endpoint.end
            ):
                continue
            source_slice = self.chunk.content[local_start:local_end]
            if unicode_normalize("NFC", source_slice) == unicode_normalize(
                "NFC", endpoint.raw_text
            ):
                return True
        return False

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
                elif not self._endpoint_has_relation_chunk_observation(endpoint):
                    errors[endpoint_name] = (
                        "Relation endpoint requires same-chunk evidence or a "
                        "compatible overlapping text observation."
                    )
        if errors:
            raise ValidationError(errors)


class CollectionRelation(ValidatedGraphModel):
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
            models.CheckConstraint(
                condition=Q(status__in=ResolutionStatus.values),
                name="kg_collection_relation_status_valid",
            ),
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

    def _raw_validation_errors(self) -> dict[str, str]:
        value = self.confidence
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            return {"confidence": "Confidence must be a finite non-boolean number."}
        return {}

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


class CollectionRelationEvidence(ValidatedGraphModel):
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
    head_mapping = models.ForeignKey(
        CollectionEntityDocumentLink,
        on_delete=models.RESTRICT,
        related_name="head_relation_evidence",
    )
    tail_mapping = models.ForeignKey(
        CollectionEntityDocumentLink,
        on_delete=models.RESTRICT,
        related_name="tail_relation_evidence",
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
        if not self.relation_id or not self.relation_mention_id:
            return

        relation = self.relation
        mention = self.relation_mention
        if relation.relation_type != mention.relation_type:
            raise ValidationError(
                {"relation_mention": "Evidence relation type must match the relation."}
            )

        errors: dict[str, str] = {}
        artifact = relation.artifact
        if artifact.scope_type != GraphArtifact.ScopeType.COLLECTION:
            errors["relation"] = "Evidence relation requires a collection artifact."
        for field_name, endpoint in (
            ("source", relation.source),
            ("target", relation.target),
        ):
            if endpoint.artifact_id != relation.artifact_id:
                errors[field_name] = "Relation endpoint artifact must match."
            elif str(endpoint.collection_id) != artifact.scope_id:
                errors[field_name] = "Relation endpoint collection must match artifact."

        for field_name, endpoint in (("head", mention.head), ("tail", mention.tail)):
            if endpoint.artifact_id != mention.artifact_id:
                errors[field_name] = (
                    "Raw endpoint artifact must match relation mention."
                )
            elif endpoint.document_id != mention.document_id:
                errors[field_name] = (
                    "Raw endpoint document must match relation mention."
                )
        if (
            mention.artifact.scope_type != GraphArtifact.ScopeType.DOCUMENT
            or mention.artifact.status != GraphArtifact.Status.ACTIVE
        ):
            errors["relation_mention"] = (
                "Evidence must come from an active document artifact."
            )

        if not errors:
            for field_name, mapping, raw_mention, collection_endpoint in (
                ("head_mapping", self.head_mapping, mention.head, relation.source),
                ("tail_mapping", self.tail_mapping, mention.tail, relation.target),
            ):
                if mapping.status != ResolutionStatus.ACTIVE:
                    errors[field_name] = "Evidence mapping must be active."
                elif mapping.collection_entity_id != collection_endpoint.pk:
                    errors[field_name] = "Evidence mapping direction is incorrect."
                elif mapping.collection_entity.artifact_id != relation.artifact_id:
                    errors[field_name] = (
                        "Evidence mapping artifact must match relation."
                    )
                elif str(mapping.collection_entity.collection_id) != artifact.scope_id:
                    errors[field_name] = (
                        "Evidence mapping collection must match relation."
                    )
                elif mapping.resolver_version != artifact.resolver_version:
                    errors[field_name] = "Evidence mapping resolver version must match."
                elif mapping.document_entity.artifact_id != mention.artifact_id:
                    errors[field_name] = "Mapped document artifact must match evidence."
                elif mapping.document_entity.document_id != mention.document_id:
                    errors[field_name] = "Mapped document must match evidence."
                elif mapping.document_entity.status != ResolutionStatus.ACTIVE:
                    errors[field_name] = "Mapped document entity must be active."
                elif collection_endpoint.status != ResolutionStatus.ACTIVE:
                    errors[field_name] = "Mapped collection entity must be active."
                elif not self._endpoint_membership_is_active(mapping, raw_mention):
                    errors[field_name] = (
                        "Raw endpoint is not actively assigned to mapping."
                    )
        if errors:
            raise ValidationError(errors)

    def _endpoint_membership_is_active(
        self,
        mapping: CollectionEntityDocumentLink,
        mention: EntityMention,
    ) -> bool:
        return DocumentEntityMention.objects.filter(
            document_entity_id=mapping.document_entity_id,
            mention_id=mention.pk,
            status=ResolutionStatus.ACTIVE,
        ).exists()
