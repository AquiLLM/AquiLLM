import re
from math import isfinite
from unicodedata import normalize as unicode_normalize

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Exists, F, OuterRef, Q

from apps.documents.models import TextChunk

from .artifacts import (
    CollectionArtifactChildModelMixin,
    CollectionArtifactChildQuerySet,
    GraphArtifact,
    ImmutableGraphQuerySet,
    ValidatedGraphModel,
)
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
        if self.artifact_id:
            artifact = self.artifact
            if artifact.scope_type != GraphArtifact.ScopeType.DOCUMENT:
                errors["artifact"] = "Relation mentions require document scope."
            elif (
                artifact.status != GraphArtifact.Status.BUILDING
                and self._state.adding
            ):
                errors["artifact"] = (
                    "Relation mentions can only be added to a building artifact."
                )
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


class CollectionRelationQuerySet(CollectionArtifactChildQuerySet):
    """Read active graph state without exposing a building shadow artifact."""

    def current(self):
        active_evidence = CollectionRelationEvidence.objects.filter(
            relation_id=OuterRef("pk"),
            status=CollectionRelationEvidence.Status.ACTIVE,
        )
        return self.annotate(_has_active_evidence=Exists(active_evidence)).filter(
            artifact__status=GraphArtifact.Status.ACTIVE,
            status=ResolutionStatus.ACTIVE,
            source__status=ResolutionStatus.ACTIVE,
            target__status=ResolutionStatus.ACTIVE,
            _has_active_evidence=True,
        )


class CollectionRelation(CollectionArtifactChildModelMixin, ValidatedGraphModel):
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
    support_count = models.PositiveIntegerField(default=1)
    confidence = models.FloatField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    _IMMUTABLE_FIELDS = (
        "artifact",
        "artifact_id",
        "source",
        "source_id",
        "relation_type",
        "target",
        "target_id",
        "status",
        "support_count",
        "confidence",
        "metadata",
        "created_at",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS

    objects = models.Manager.from_queryset(CollectionRelationQuerySet)()

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
                condition=Q(support_count__gte=1),
                name="kg_collection_relation_support",
            ),
            models.CheckConstraint(
                condition=~Q(status=ResolutionStatus.ACTIVE)
                | Q(support_count__gte=1),
                name="kg_active_collection_relation_supported",
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
            models.Index(
                fields=["artifact", "target", "source", "relation_type"],
                name="kg_collection_relation_rev",
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
            if self.status == self.Status.ACTIVE:
                if self.source.status != ResolutionStatus.ACTIVE:
                    errors["source"] = "Active relation source must be active."
                if self.target.status != ResolutionStatus.ACTIVE:
                    errors["target"] = "Active relation target must be active."
        if self.artifact_id:
            artifact = self.artifact
            if artifact.scope_type != GraphArtifact.ScopeType.COLLECTION:
                errors["artifact"] = "Collection relations require collection scope."
            elif (
                artifact.status != GraphArtifact.Status.BUILDING
                and self._state.adding
            ):
                errors["artifact"] = "Collection relations require a building artifact."
        if self.support_count < 1:
            errors["support_count"] = "Collection relations require real evidence."
        if errors:
            raise ValidationError(errors)

class CollectionRelationEvidence(
    CollectionArtifactChildModelMixin, ValidatedGraphModel
):
    """One retained supporting extraction for a collection relation."""

    class Status(models.TextChoices):
        ACTIVE = "active", "Promoted"
        SUPPRESSED = "suppressed", "Suppressed"
        REJECTED = "rejected", "Rejected"

    class Orientation(models.TextChoices):
        HEAD_TO_TAIL = "head_to_tail", "Raw head maps to relation source"
        TAIL_TO_HEAD = "tail_to_head", "Raw tail maps to relation source"

    artifact = models.ForeignKey(
        GraphArtifact,
        on_delete=models.CASCADE,
        related_name="collection_relation_evidence",
    )

    relation = models.ForeignKey(
        CollectionRelation,
        on_delete=models.RESTRICT,
        related_name="evidence",
        null=True,
        blank=True,
    )
    relation_mention = models.ForeignKey(
        RelationMention,
        on_delete=models.PROTECT,
        related_name="collection_evidence_links",
    )
    head_mapping = models.ForeignKey(
        CollectionEntityDocumentLink,
        on_delete=models.RESTRICT,
        related_name="head_relation_evidence",
        null=True,
        blank=True,
    )
    tail_mapping = models.ForeignKey(
        CollectionEntityDocumentLink,
        on_delete=models.RESTRICT,
        related_name="tail_relation_evidence",
        null=True,
        blank=True,
    )
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    reason = models.CharField(
        max_length=128, default="ontology_valid_supported_evidence"
    )
    orientation = models.CharField(
        max_length=16,
        choices=Orientation.choices,
        default=Orientation.HEAD_TO_TAIL,
    )
    ontology_checksum = models.CharField(max_length=64, editable=False)
    assembly_config_checksum = models.CharField(max_length=64, editable=False)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    _IMMUTABLE_FIELDS = (
        "artifact",
        "artifact_id",
        "relation",
        "relation_id",
        "relation_mention",
        "relation_mention_id",
        "head_mapping",
        "head_mapping_id",
        "tail_mapping",
        "tail_mapping_id",
        "status",
        "reason",
        "orientation",
        "ontology_checksum",
        "assembly_config_checksum",
        "metadata",
        "created_at",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS

    objects = models.Manager.from_queryset(CollectionArtifactChildQuerySet)()

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["artifact", "relation_mention"],
                name="kg_relation_evidence_unique",
            ),
            models.CheckConstraint(
                condition=(
                    Q(
                        status="active",
                        relation__isnull=False,
                        head_mapping__isnull=False,
                        tail_mapping__isnull=False,
                    )
                    | Q(
                        status__in=("suppressed", "rejected"),
                        relation__isnull=True,
                    )
                ),
                name="kg_relation_evidence_decision_valid",
            ),
            models.CheckConstraint(
                condition=Q(ontology_checksum__regex=r"^[0-9a-f]{64}$"),
                name="kg_relation_evidence_ontology_hash",
            ),
            models.CheckConstraint(
                condition=Q(assembly_config_checksum__regex=r"^[0-9a-f]{64}$"),
                name="kg_relation_evidence_config_hash",
            ),
        ]

    def clean(self):
        super().clean()
        if not self.relation_mention_id:
            return

        mention = self.relation_mention
        errors: dict[str, str] = {}
        relation = self.relation if self.relation_id else None
        if relation is not None and relation.relation_type != mention.relation_type:
            errors["relation_mention"] = (
                "Evidence relation type must match the relation."
            )
        artifact = None
        if self.artifact_id:
            artifact = self.artifact
        elif (
            relation is not None
            and relation.artifact_id
            and "artifact" in relation._state.fields_cache
        ):
            artifact = relation.artifact
        if (
            artifact is not None
            and artifact.scope_type != GraphArtifact.ScopeType.COLLECTION
        ):
            errors["artifact"] = "Evidence requires a collection artifact."
        elif (
            artifact is not None
            and artifact.status != GraphArtifact.Status.BUILDING
            and self._state.adding
        ):
            errors["artifact"] = "Evidence requires a building artifact."
        if self.status == self.Status.ACTIVE:
            if not self.relation_id:
                errors["relation"] = "Promoted evidence requires a relation."
            elif self.artifact_id and relation.artifact_id != self.artifact_id:
                errors["relation"] = "Evidence relation artifact must match."
            if not self.head_mapping_id or not self.tail_mapping_id:
                errors["head_mapping"] = "Promoted evidence requires both mappings."
        elif self.relation_id:
            errors["relation"] = "Rejected evidence cannot promote a relation."
        if not self.reason:
            errors["reason"] = "Evidence decisions require an audit reason."
        if self.artifact_id:
            for field_name in ("ontology_checksum", "assembly_config_checksum"):
                value = getattr(self, field_name)
                if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
                    errors[field_name] = (
                        "Evidence identity must be a SHA-256 checksum."
                    )
            if (
                artifact is not None
                and self.ontology_checksum != artifact.ontology_checksum
            ):
                errors["ontology_checksum"] = (
                    "Evidence ontology checksum must match its artifact."
                )
            if (
                artifact is not None
                and self.assembly_config_checksum
                != artifact.assembly_config_checksum
            ):
                errors["assembly_config_checksum"] = (
                    "Evidence assembly checksum must match its artifact."
                )

        if errors:
            raise ValidationError(errors)

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

        if artifact is not None and (not errors or self.status != self.Status.ACTIVE):
            head_endpoint = None
            tail_endpoint = None
            if relation is not None:
                if self.orientation == self.Orientation.HEAD_TO_TAIL:
                    head_endpoint, tail_endpoint = relation.source, relation.target
                else:
                    head_endpoint, tail_endpoint = relation.target, relation.source
            mapping_inputs = (
                (
                    "head_mapping",
                    self.head_mapping if self.head_mapping_id else None,
                    mention.head,
                    head_endpoint,
                ),
                (
                    "tail_mapping",
                    self.tail_mapping if self.tail_mapping_id else None,
                    mention.tail,
                    tail_endpoint,
                ),
            )
            for field_name, mapping, raw_mention, collection_endpoint in mapping_inputs:
                if mapping is None:
                    continue
                if mapping.status != ResolutionStatus.ACTIVE:
                    errors[field_name] = "Evidence mapping must be active."
                elif mapping.outcome != CollectionEntityDocumentLink.Outcome.AUTOMATIC:
                    errors[field_name] = "Evidence requires an automatic mapping."
                elif (
                    collection_endpoint is not None
                    and mapping.collection_entity_id != collection_endpoint.pk
                ):
                    errors[field_name] = "Evidence mapping direction is incorrect."
                elif (
                    mapping.artifact_id != artifact.pk
                    or mapping.collection_entity.artifact_id != artifact.pk
                ):
                    errors[field_name] = (
                        "Evidence mapping artifact must match relation."
                    )
                elif (
                    not mapping.manifest_input_id
                    or mapping.manifest_input.artifact_id != artifact.pk
                    or mapping.manifest_input.document_artifact_id
                    != mention.artifact_id
                    or mapping.manifest_input.document_id != mention.document_id
                    or str(mapping.manifest_input.collection_id) != artifact.scope_id
                ):
                    errors[field_name] = (
                        "Evidence mapping manifest must match the exact source."
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
                elif (
                    collection_endpoint is not None
                    and collection_endpoint.status != ResolutionStatus.ACTIVE
                ):
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
