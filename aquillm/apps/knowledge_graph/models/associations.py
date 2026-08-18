from math import isfinite

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q

from .artifacts import GraphArtifact, ImmutableGraphQuerySet, ValidatedGraphModel
from .entities import (
    CanonicalEntity,
    CollectionEntity,
    DocumentEntity,
    ResolutionStatus,
)
from .inputs import CollectionArtifactInput


class CollectionEntityDocumentLink(ValidatedGraphModel):
    """Versioned resolver decision linking document and collection nodes."""

    Status = ResolutionStatus

    class Outcome(models.TextChoices):
        AUTOMATIC = "automatic", "Automatic"
        CANDIDATE = "candidate", "Candidate"
        REJECTED = "rejected", "Rejected"

    artifact = models.ForeignKey(
        GraphArtifact,
        on_delete=models.CASCADE,
        related_name="collection_entity_links",
    )
    manifest_input = models.ForeignKey(
        CollectionArtifactInput,
        on_delete=models.RESTRICT,
        related_name="entity_links",
    )

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
    identifier_score = models.FloatField(null=True, blank=True)
    alias_score = models.FloatField(null=True, blank=True)
    embedding_similarity = models.FloatField(null=True, blank=True)
    neighborhood_agreement = models.FloatField(null=True, blank=True)
    method = models.CharField(max_length=64)
    resolver_version = models.CharField(max_length=128)
    outcome = models.CharField(max_length=16, choices=Outcome.choices)
    candidate_rank = models.PositiveIntegerField(null=True, blank=True)
    decision_checksum = models.CharField(max_length=64, editable=False)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    reason = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    _IMMUTABLE_FIELDS = (
        "artifact",
        "artifact_id",
        "manifest_input",
        "manifest_input_id",
        "document_entity",
        "document_entity_id",
        "collection_entity",
        "collection_entity_id",
        "score",
        "identifier_score",
        "alias_score",
        "embedding_similarity",
        "neighborhood_agreement",
        "method",
        "resolver_version",
        "outcome",
        "candidate_rank",
        "decision_checksum",
        "status",
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
                name="kg_doc_collection_link_status_valid",
            ),
            models.CheckConstraint(
                condition=Q(score__gte=0) & Q(score__lte=1),
                name="kg_doc_collection_score_range",
            ),
            *(
                models.CheckConstraint(
                    condition=Q(**{f"{field_name}__isnull": True})
                    | (Q(**{f"{field_name}__gte": 0}) & Q(**{f"{field_name}__lte": 1})),
                    name=constraint_name,
                )
                for field_name, constraint_name in (
                    ("identifier_score", "kg_doc_link_identifier_score"),
                    ("alias_score", "kg_doc_link_alias_score"),
                    ("embedding_similarity", "kg_doc_link_embedding_score"),
                    ("neighborhood_agreement", "kg_doc_link_neighbor_score"),
                )
            ),
            models.CheckConstraint(
                condition=Q(outcome__in=("automatic", "candidate", "rejected")),
                name="kg_doc_collection_outcome_valid",
            ),
            models.CheckConstraint(
                condition=(
                    Q(outcome="automatic", candidate_rank__isnull=True)
                    | Q(outcome="candidate", candidate_rank__isnull=False)
                    | Q(outcome="rejected")
                ),
                name="kg_doc_collection_candidate_rank",
            ),
            models.CheckConstraint(
                condition=(
                    Q(outcome="automatic", status=ResolutionStatus.ACTIVE)
                    | Q(outcome="candidate", status=ResolutionStatus.SUPPRESSED)
                    | Q(outcome="rejected", status=ResolutionStatus.REJECTED)
                ),
                name="kg_doc_collection_outcome_status",
            ),
            models.CheckConstraint(
                condition=Q(decision_checksum__regex=r"^[0-9a-f]{64}$"),
                name="kg_doc_collection_decision_hash",
            ),
            models.UniqueConstraint(
                fields=[
                    "artifact",
                    "document_entity",
                    "collection_entity",
                    "resolver_version",
                ],
                name="kg_doc_collection_entity_link_unique",
            ),
            models.UniqueConstraint(
                fields=["artifact", "document_entity"],
                condition=Q(outcome="automatic"),
                name="kg_one_auto_collection_assignment",
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
            models.Index(
                fields=["artifact", "outcome", "document_entity"],
                name="kg_doc_link_outcome_idx",
            ),
        ]

    def _raw_validation_errors(self) -> dict[str, str]:
        value = self.score
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            return {"score": "Score must be a finite non-boolean number."}
        errors: dict[str, str] = {}
        for field_name in (
            "identifier_score",
            "alias_score",
            "embedding_similarity",
            "neighborhood_agreement",
        ):
            component = getattr(self, field_name)
            if component is not None and (
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not isfinite(component)
            ):
                errors[field_name] = "Score must be a finite non-boolean number."
        return errors

    def clean(self):
        super().clean()
        errors: dict[str, str] = {}
        if self.artifact_id and self.collection_entity_id:
            if self.collection_entity.artifact_id != self.artifact_id:
                errors["collection_entity"] = (
                    "Collection entity must belong to the destination artifact."
                )
            elif str(self.collection_entity.collection_id) != self.artifact.scope_id:
                errors["collection_entity"] = (
                    "Collection entity must match destination collection scope."
                )
        if self.manifest_input_id and self.artifact_id:
            if self.manifest_input.artifact_id != self.artifact_id:
                errors["manifest_input"] = (
                    "Manifest input must belong to the destination artifact."
                )
        if self.manifest_input_id and self.document_entity_id:
            if (
                self.document_entity.artifact_id
                != self.manifest_input.document_artifact_id
            ):
                errors["document_entity"] = (
                    "Document entity must belong to the manifest source artifact."
                )
            elif self.document_entity.document_id != self.manifest_input.document_id:
                errors["document_entity"] = (
                    "Document entity must match the manifest document."
                )
        if self.manifest_input_id and self.collection_entity_id:
            if (
                self.collection_entity.collection_id
                != self.manifest_input.collection_id
            ):
                errors["collection_entity"] = (
                    "Collection entity must match the manifest collection."
                )
        if self.artifact_id:
            if self.artifact.status != GraphArtifact.Status.BUILDING and not self.pk:
                errors["artifact"] = "Resolution links require a building artifact."
            if self.resolver_version != self.artifact.resolver_version:
                errors["resolver_version"] = (
                    "Resolution link version must match destination artifact."
                )
        expected_status = {
            self.Outcome.AUTOMATIC: self.Status.ACTIVE,
            self.Outcome.CANDIDATE: self.Status.SUPPRESSED,
            self.Outcome.REJECTED: self.Status.REJECTED,
        }.get(self.outcome)
        if expected_status is not None and self.status != expected_status:
            errors["status"] = "Status must correspond to resolution outcome."
        if not self.reason:
            errors["reason"] = "Resolution links require an audit reason."
        if self.candidate_rank is not None and self.candidate_rank < 1:
            errors["candidate_rank"] = "Candidate rank must be positive."
        if self.outcome == self.Outcome.AUTOMATIC and self.candidate_rank is not None:
            errors["candidate_rank"] = "Automatic assignments cannot have a rank."
        if self.outcome == self.Outcome.CANDIDATE and self.candidate_rank is None:
            errors["candidate_rank"] = "Candidate assignments require a rank."
        for field_name in (
            "score",
            "identifier_score",
            "alias_score",
            "embedding_similarity",
            "neighborhood_agreement",
        ):
            component = getattr(self, field_name)
            if component is not None and not 0 <= component <= 1:
                errors[field_name] = "Score must be in [0, 1]."
        if errors:
            raise ValidationError(errors)


class CanonicalEntityLink(ValidatedGraphModel):
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
                condition=Q(status__in=ResolutionStatus.values),
                name="kg_canonical_link_status_valid",
            ),
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

    def _raw_validation_errors(self) -> dict[str, str]:
        value = self.score
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            return {"score": "Score must be a finite non-boolean number."}
        return {}
