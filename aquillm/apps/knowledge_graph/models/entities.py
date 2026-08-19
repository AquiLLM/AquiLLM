from __future__ import annotations

import re
from math import isfinite

from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import models
from django.db.models import F, Q
from pgvector.django import VectorField

from apps.collections.models import Collection
from apps.documents.models import TextChunk

from .artifacts import (
    CollectionArtifactChildModelMixin,
    CollectionArtifactChildQuerySet,
    GraphArtifact,
    ImmutableGraphQuerySet,
    ValidatedGraphModel,
)


class ResolutionStatus(models.TextChoices):
    ACTIVE = "active", "Active"
    SUPPRESSED = "suppressed", "Suppressed"
    REJECTED = "rejected", "Rejected"
    SUPERSEDED = "superseded", "Superseded"


class EntityMention(ValidatedGraphModel):
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
    normalized_text = models.CharField(max_length=512)
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
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    _IMMUTABLE_FIELDS = (
        "artifact",
        "artifact_id",
        "document_id",
        "chunk",
        "chunk_id",
        "start",
        "end",
        "position_basis",
        "raw_text",
        "normalized_text",
        "entity_type",
        "extraction_confidence",
        "content_object_type",
        "content_object_type_id",
        "content_object_id",
        "metadata",
        "created_at",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS

    objects = models.Manager.from_queryset(ImmutableGraphQuerySet)()

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
                condition=Q(position_basis__in=("document_global", "chunk_content")),
                name="kg_mention_position_basis_valid",
            ),
            models.CheckConstraint(
                condition=(
                    Q(
                        position_basis="document_global",
                        content_object_type__isnull=True,
                        content_object_id__isnull=True,
                    )
                    | Q(
                        position_basis="chunk_content",
                        content_object_type__isnull=False,
                        content_object_id__isnull=False,
                    )
                ),
                name="kg_mention_basis_provenance",
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
            models.Index(
                fields=["content_object_type", "content_object_id"],
                name="kg_mention_content_obj_idx",
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

    def _resolve_image_content_object(self):
        try:
            target = self.content_object_type.get_object_for_this_type(
                id=self.content_object_id
            )
            document = self.chunk.document
        except (ObjectDoesNotExist, ValidationError):
            return None, None
        return target, document

    def clean(self):
        super().clean()
        errors: dict[str, str] = {}
        if self.start < 0 or self.end <= self.start:
            errors["end"] = "Mention spans must be nonnegative and nonempty."
        if not 0 <= self.extraction_confidence <= 1:
            errors["extraction_confidence"] = "Confidence must be in [0, 1]."
        chunk = self.chunk
        artifact = self.artifact
        if artifact.scope_type != GraphArtifact.ScopeType.DOCUMENT:
            errors["artifact"] = "Entity mentions require a document artifact."
        elif artifact.scope_id != str(self.document_id):
            errors["artifact"] = "Entity mention document must match artifact scope."
        elif artifact.status not in {
            GraphArtifact.Status.BUILDING,
            GraphArtifact.Status.ACTIVE,
        }:
            errors["artifact"] = "Entity mention artifact must be building or active."
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
            elif self.position_basis == self.PositionBasis.CHUNK_CONTENT:
                target, document = self._resolve_image_content_object()
                if target is None or document is None:
                    errors["content_object_id"] = (
                        "Image provenance must resolve an existing document object."
                    )
                elif (
                    target._meta.app_label,
                    target._meta.model_name,
                ) != (
                    document._meta.app_label,
                    document._meta.model_name,
                ):
                    errors["content_object_type"] = (
                        "Image provenance must use the exact document subtype."
                    )
        if errors:
            raise ValidationError(errors)


class DocumentEntity(ValidatedGraphModel):
    """Entity resolved within a document artifact."""

    Status = ResolutionStatus

    artifact = models.ForeignKey(
        GraphArtifact,
        on_delete=models.CASCADE,
        related_name="document_entities",
    )
    document_id = models.UUIDField()
    cluster_key = models.CharField(max_length=64, editable=False)
    label = models.TextField()
    normalized_label = models.CharField(max_length=512)
    version_signature = models.CharField(
        max_length=128, blank=True, default="", editable=False
    )
    resolution_confidence = models.FloatField(default=1.0)
    entity_type = models.CharField(max_length=128)
    identifier = models.CharField(max_length=255, blank=True, default="")
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    _IMMUTABLE_FIELDS = (
        "artifact",
        "artifact_id",
        "document_id",
        "cluster_key",
        "label",
        "identifier",
        "normalized_label",
        "version_signature",
        "resolution_confidence",
        "entity_type",
        "metadata",
        "created_at",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS

    objects = models.Manager.from_queryset(ImmutableGraphQuerySet)()

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(status__in=ResolutionStatus.values),
                name="kg_document_entity_status_valid",
            ),
            models.UniqueConstraint(
                fields=[
                    "artifact",
                    "entity_type",
                    "identifier",
                    "version_signature",
                ],
                condition=~Q(identifier=""),
                name="kg_document_entity_identifier_unique",
            ),
            models.UniqueConstraint(
                fields=["artifact", "cluster_key"],
                name="kg_document_entity_cluster_unique",
            ),
            models.CheckConstraint(
                condition=Q(identifier="") | ~Q(identifier__regex=r"^\s+$"),
                name="kg_document_identifier_not_ws",
            ),
            models.CheckConstraint(
                condition=Q(cluster_key__regex=r"^[0-9a-f]{64}$"),
                name="kg_document_cluster_key_valid",
            ),
            models.CheckConstraint(
                condition=Q(version_signature="")
                | Q(version_signature__regex=(r"^[a-z0-9][a-z0-9.+:/_-]*$")),
                name="kg_document_version_signature_valid",
            ),
            models.CheckConstraint(
                condition=Q(resolution_confidence__gte=0)
                & Q(resolution_confidence__lte=1),
                name="kg_document_resolution_conf_range",
            ),
        ]
        indexes = [
            models.Index(
                fields=["document_id", "entity_type", "normalized_label"],
                name="kg_doc_entity_lookup_idx",
            )
        ]

    def _normalize_identifier(self) -> None:
        original_identifier = self.identifier
        self.identifier = original_identifier.strip()
        if original_identifier and not self.identifier:
            raise ValidationError(
                {"identifier": "Identifier cannot be whitespace-only."}
            )

    def _raw_validation_errors(self) -> dict[str, str]:
        value = self.resolution_confidence
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            return {
                "resolution_confidence": (
                    "Resolution confidence must be a finite non-boolean number."
                )
            }
        return {}

    def prepare_for_persistence(self) -> None:
        self._normalize_identifier()

    def clean(self):
        super().clean()
        self._normalize_identifier()
        if not re.fullmatch(r"[0-9a-f]{64}", self.cluster_key or ""):
            raise ValidationError(
                {"cluster_key": "Cluster key must be a lowercase SHA-256 digest."}
            )
        if not isinstance(self.version_signature, str) or (
            self.version_signature
            and (
                len(self.version_signature) > 128
                or not re.fullmatch(r"[a-z0-9][a-z0-9.+:/_-]*", self.version_signature)
            )
        ):
            raise ValidationError(
                {
                    "version_signature": (
                        "Version signature must use canonical lower-ASCII tokens."
                    )
                }
            )
        if (
            self.artifact_id
            and self.artifact.scope_type != GraphArtifact.ScopeType.DOCUMENT
        ):
            raise ValidationError(
                {"artifact": "Document entities require a document artifact."}
            )
        if self.artifact_id and str(self.document_id) != self.artifact.scope_id:
            raise ValidationError(
                {"document_id": "Document entity must match artifact scope."}
            )
        if not 0 <= self.resolution_confidence <= 1:
            raise ValidationError(
                {"resolution_confidence": "Resolution confidence must be in [0, 1]."}
            )


class DocumentEntityMention(ValidatedGraphModel):
    """Auditable assignment of one mention to one document entity."""

    Status = ResolutionStatus

    class Method(models.TextChoices):
        ROOT = "root", "Cluster root"
        STABLE_IDENTIFIER = "stable_identifier", "Stable identifier"
        DEFINED_ACRONYM = "defined_acronym", "Defined acronym"
        ONTOLOGY_ALIAS = "ontology_alias", "Ontology alias"
        NORMALIZED_NAME = "normalized_name", "Normalized name"
        SINGLETON = "singleton", "Singleton"

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
    method = models.CharField(max_length=64, choices=Method.choices)
    resolver_version = models.CharField(max_length=128)
    parent_mention_id = models.CharField(
        max_length=128, blank=True, default="", editable=False
    )
    reason = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    _IMMUTABLE_FIELDS = (
        "document_entity",
        "document_entity_id",
        "mention",
        "mention_id",
        "method",
        "resolver_version",
        "parent_mention_id",
        "reason",
        "metadata",
        "created_at",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS

    objects = models.Manager.from_queryset(ImmutableGraphQuerySet)()

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(status__in=ResolutionStatus.values),
                name="kg_document_mention_status_valid",
            ),
            models.CheckConstraint(
                condition=Q(
                    method__in=(
                        "root",
                        "stable_identifier",
                        "defined_acronym",
                        "ontology_alias",
                        "normalized_name",
                        "singleton",
                    )
                ),
                name="kg_document_mention_method_valid",
            ),
            models.CheckConstraint(
                condition=~Q(resolver_version=""),
                name="kg_document_mention_resolver_nonempty",
            ),
            models.CheckConstraint(
                condition=(Q(method__in=("root", "singleton"), parent_mention_id=""))
                | (~Q(method__in=("root", "singleton")) & ~Q(parent_mention_id="")),
                name="kg_document_mention_parent_valid",
            ),
        ]
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
            if self.resolver_version != self.document_entity.artifact.resolver_version:
                raise ValidationError(
                    {
                        "resolver_version": (
                            "Mention link resolver version must match its artifact."
                        )
                    }
                )
            if self.parent_mention_id == str(self.mention_id):
                raise ValidationError(
                    {"parent_mention_id": "A mention link cannot parent itself."}
                )


class CollectionEntityQuerySet(CollectionArtifactChildQuerySet):
    """Expose only entities belonging to the active collection graph."""

    def current(self):
        return self.filter(
            artifact__status=GraphArtifact.Status.ACTIVE,
            status=ResolutionStatus.ACTIVE,
        )


class CollectionEntity(CollectionArtifactChildModelMixin, ValidatedGraphModel):
    """Collection-scoped resolved entity with optional retrieval embedding."""

    Status = ResolutionStatus

    artifact = models.ForeignKey(
        GraphArtifact,
        on_delete=models.CASCADE,
        related_name="collection_entities",
    )
    collection = models.ForeignKey(
        Collection,
        on_delete=models.DO_NOTHING,
        related_name="knowledge_graph_entities",
    )
    cluster_key = models.CharField(max_length=64, editable=False)
    label = models.TextField()
    normalized_label = models.CharField(max_length=512)
    version_signature = models.CharField(
        max_length=128, blank=True, default="", editable=False
    )
    entity_type = models.CharField(max_length=128)
    identifier = models.CharField(max_length=255, blank=True, default="")
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    extraction_confidence = models.FloatField()
    resolution_confidence = models.FloatField()
    retrieval_utility = models.FloatField()
    promotion_confidence = models.FloatField(null=True, blank=True)
    filter_reason = models.CharField(max_length=128, blank=True, default="")
    embedding_model_signature = models.CharField(
        max_length=512, blank=True, default="", editable=False
    )
    embedding_input_hash = models.CharField(
        max_length=64, blank=True, default="", editable=False
    )
    embedding = VectorField(dimensions=1024, blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    _IMMUTABLE_FIELDS = (
        "artifact",
        "artifact_id",
        "collection",
        "collection_id",
        "cluster_key",
        "label",
        "identifier",
        "normalized_label",
        "version_signature",
        "entity_type",
        "status",
        "extraction_confidence",
        "resolution_confidence",
        "retrieval_utility",
        "promotion_confidence",
        "filter_reason",
        "embedding_model_signature",
        "embedding_input_hash",
        "embedding",
        "metadata",
        "created_at",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS

    objects = models.Manager.from_queryset(CollectionEntityQuerySet)()

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(status__in=ResolutionStatus.values),
                name="kg_collection_entity_status_valid",
            ),
            models.UniqueConstraint(
                fields=[
                    "artifact",
                    "entity_type",
                    "identifier",
                    "version_signature",
                ],
                condition=~Q(identifier=""),
                name="kg_collection_entity_identifier_unique",
            ),
            models.UniqueConstraint(
                fields=["artifact", "cluster_key"],
                name="kg_collection_entity_cluster_unique",
            ),
            models.CheckConstraint(
                condition=Q(identifier="") | ~Q(identifier__regex=r"^\s+$"),
                name="kg_collection_identifier_not_ws",
            ),
            models.CheckConstraint(
                condition=Q(cluster_key__regex=r"^[0-9a-f]{64}$"),
                name="kg_collection_cluster_key_valid",
            ),
            models.CheckConstraint(
                condition=Q(version_signature="")
                | Q(version_signature__regex=r"^[a-z0-9][a-z0-9.+:/_-]*$"),
                name="kg_collection_version_signature_valid",
            ),
            *(
                models.CheckConstraint(
                    condition=Q(**{f"{field_name}__gte": 0})
                    & Q(**{f"{field_name}__lte": 1}),
                    name=constraint_name,
                )
                for field_name, constraint_name in (
                    ("extraction_confidence", "kg_collection_extract_conf_range"),
                    ("resolution_confidence", "kg_collection_resolve_conf_range"),
                    ("retrieval_utility", "kg_collection_utility_range"),
                )
            ),
            models.CheckConstraint(
                condition=Q(promotion_confidence__isnull=True)
                | (Q(promotion_confidence__gte=0) & Q(promotion_confidence__lte=1)),
                name="kg_collection_promotion_conf_range",
            ),
            models.CheckConstraint(
                condition=(
                    Q(
                        embedding__isnull=True,
                        embedding_model_signature="",
                        embedding_input_hash="",
                    )
                    | (
                        Q(embedding__isnull=False)
                        & ~Q(embedding_model_signature="")
                        & Q(embedding_input_hash__regex=r"^[0-9a-f]{64}$")
                    )
                ),
                name="kg_collection_embedding_audit_complete",
            ),
        ]
        indexes = [
            models.Index(
                fields=[
                    "artifact",
                    "collection",
                    "entity_type",
                    "normalized_label",
                ],
                name="kg_collection_entity_lookup",
            )
        ]

    def _normalize_identifier(self) -> None:
        original_identifier = self.identifier
        self.identifier = original_identifier.strip()
        if original_identifier and not self.identifier:
            raise ValidationError(
                {"identifier": "Identifier cannot be whitespace-only."}
            )

    def prepare_for_persistence(self) -> None:
        self._normalize_identifier()

    def _raw_validation_errors(self) -> dict[str, str]:
        errors: dict[str, str] = {}
        for field_name in (
            "extraction_confidence",
            "resolution_confidence",
            "retrieval_utility",
            "promotion_confidence",
        ):
            value = getattr(self, field_name)
            if field_name == "promotion_confidence" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                errors[field_name] = "Score must be a finite non-boolean number."
        return errors

    def clean(self):
        super().clean()
        self._normalize_identifier()
        errors: dict[str, str] = {}
        if not re.fullmatch(r"[0-9a-f]{64}", self.cluster_key or ""):
            errors["cluster_key"] = "Cluster key must be a lowercase SHA-256 digest."
        if self.version_signature and not re.fullmatch(
            r"[a-z0-9][a-z0-9.+:/_-]*", self.version_signature
        ):
            errors["version_signature"] = "Version signature is not canonical."
        for field_name in (
            "extraction_confidence",
            "resolution_confidence",
            "retrieval_utility",
            "promotion_confidence",
        ):
            value = getattr(self, field_name)
            if field_name == "promotion_confidence" and value is None:
                continue
            if not 0 <= value <= 1:
                errors[field_name] = "Score must be in [0, 1]."
        if (
            self.artifact_id
            and self.artifact.scope_type != GraphArtifact.ScopeType.COLLECTION
        ):
            errors["artifact"] = "Collection entities require a collection artifact."
        if self.artifact_id and str(self.collection_id) != self.artifact.scope_id:
            errors["collection"] = "Collection entity must match artifact scope."
        if (
            self.artifact_id
            and self.artifact.status != GraphArtifact.Status.BUILDING
            and self._state.adding
        ):
            errors["artifact"] = "Collection entities require a building artifact."
        if self.embedding is None:
            if self.embedding_model_signature or self.embedding_input_hash:
                errors["embedding"] = "Missing embeddings cannot carry audit identity."
        else:
            try:
                vector = tuple(float(value) for value in self.embedding)
            except (TypeError, ValueError, OverflowError):
                vector = ()
            if len(vector) != 1024 or any(not isfinite(value) for value in vector):
                errors["embedding"] = "Embedding must contain 1024 finite dimensions."
            if (
                self.embedding_model_signature
                != self.artifact.embedding_model_signature
            ):
                errors["embedding_model_signature"] = (
                    "Entity embedding signature must match its artifact."
                )
            if not re.fullmatch(r"[0-9a-f]{64}", self.embedding_input_hash or ""):
                errors["embedding_input_hash"] = (
                    "Embedding input hash must be a lowercase SHA-256 digest."
                )
        if self.status in {self.Status.SUPPRESSED, self.Status.REJECTED} and not (
            self.filter_reason
        ):
            errors["filter_reason"] = "Filtered entities require a reason code."
        if errors:
            raise ValidationError(errors)


class CanonicalEntityQuerySet(ImmutableGraphQuerySet):
    """Expose only live registry rows for one exact resolver version."""

    def current(self, *, resolver_version: str):
        if (
            type(resolver_version) is not str
            or not resolver_version
            or resolver_version != resolver_version.strip()
        ):
            raise ValueError("resolver_version must be a nonempty exact string")
        return self.filter(
            resolver_version=resolver_version,
            status=ResolutionStatus.ACTIVE,
        )


class CanonicalEntity(ValidatedGraphModel):
    """Internal cross-collection identity without evidence or access grants."""

    Status = ResolutionStatus

    identity_key = models.CharField(max_length=64, editable=False)
    resolver_version = models.CharField(max_length=128, editable=False)
    label = models.TextField()
    normalized_label = models.CharField(max_length=512)
    entity_type = models.CharField(max_length=128)
    version_signature = models.CharField(
        max_length=128, blank=True, default="", editable=False
    )
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    embedding = VectorField(dimensions=1024, blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    _IMMUTABLE_FIELDS = (
        "identity_key",
        "resolver_version",
        "label",
        "normalized_label",
        "entity_type",
        "version_signature",
        "status",
        "embedding",
        "metadata",
        "created_at",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS

    objects = models.Manager.from_queryset(CanonicalEntityQuerySet)()

    @classmethod
    def _transition_registry_status_locked(
        cls,
        primary_keys: tuple[int, ...],
        *,
        target: str,
        using: str = "default",
    ) -> int:
        """Internal lifecycle transition for already registry-locked rows."""

        from django.db import connections

        if not connections[using].in_atomic_block:
            raise RuntimeError("canonical status transitions require an atomic block")
        if target not in {cls.Status.ACTIVE, cls.Status.SUPERSEDED}:
            raise ValueError("canonical registry target status is invalid")
        if type(primary_keys) is not tuple or any(
            type(value) is not int or value < 1 for value in primary_keys
        ):
            raise ValueError("canonical registry primary keys are invalid")
        if primary_keys != tuple(sorted(set(primary_keys))):
            raise ValueError(
                "canonical registry primary keys must be sorted and unique"
            )
        changed = 0
        for start in range(0, len(primary_keys), 5_000):
            query = cls.objects.using(using).filter(
                pk__in=primary_keys[start : start + 5_000],
                status__in=(cls.Status.ACTIVE, cls.Status.SUPERSEDED),
            )
            changed += models.QuerySet.update(query, status=target)
        return changed

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(status__in=ResolutionStatus.values),
                name="kg_canonical_entity_status_valid",
            ),
            models.CheckConstraint(
                condition=Q(identity_key__regex=r"^[0-9a-f]{64}$"),
                name="kg_canonical_identity_key_valid",
            ),
            models.UniqueConstraint(
                fields=["resolver_version", "identity_key"],
                name="kg_canonical_identity_unique",
            ),
            models.CheckConstraint(
                condition=~Q(resolver_version=""),
                name="kg_canonical_resolver_nonempty",
            ),
            models.CheckConstraint(
                condition=Q(version_signature="")
                | Q(version_signature__regex=r"^[a-z0-9][a-z0-9.+:/_-]*$"),
                name="kg_canonical_version_signature_valid",
            ),
            models.CheckConstraint(
                condition=~Q(resolver_version="canonical-resolution-v1")
                | Q(embedding__isnull=True),
                name="kg_canonical_v1_embedding_null",
            ),
        ]
        indexes = [
            models.Index(
                fields=["entity_type", "normalized_label"],
                name="kg_canonical_entity_lookup",
            ),
            models.Index(
                fields=["resolver_version", "status", "entity_type"],
                name="kg_can_entity_res_type_idx",
            ),
        ]

    def clean(self):
        super().clean()
        errors: dict[str, str] = {}
        if not re.fullmatch(r"[0-9a-f]{64}", self.identity_key or ""):
            errors["identity_key"] = "Identity key must be a lowercase SHA-256 digest."
        if (
            type(self.resolver_version) is not str
            or not self.resolver_version
            or self.resolver_version != self.resolver_version.strip()
            or len(self.resolver_version) > 128
            or "\x00" in self.resolver_version
        ):
            errors["resolver_version"] = (
                "Resolver version must be a bounded exact string."
            )
        if type(self.metadata) is not dict:
            errors["metadata"] = "Canonical metadata must be an exact mapping."
        if self.version_signature and not re.fullmatch(
            r"[a-z0-9][a-z0-9.+:/_-]*", self.version_signature
        ):
            errors["version_signature"] = "Version signature is not canonical."
        if self.embedding is not None:
            errors["embedding"] = (
                "Canonical registry embeddings are not an audited v1 identity signal."
            )
        if errors:
            raise ValidationError(errors)
