import json
import re
from hashlib import sha256
from math import isfinite

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import F, Q

from .artifacts import (
    CollectionArtifactChildModelMixin,
    CollectionArtifactChildQuerySet,
    GraphArtifact,
    ImmutableGraphQuerySet,
    ValidatedGraphModel,
)
from .entities import (
    CanonicalEntity,
    CollectionEntity,
    DocumentEntity,
    ResolutionStatus,
)
from .inputs import CollectionArtifactInput


def _current_link_filters(prefix: str = "") -> dict[str, object]:
    """Return one fail-closed automatic mapping path for current-state reads."""

    path = f"{prefix}__" if prefix else ""
    return {
        f"{path}artifact__status": GraphArtifact.Status.ACTIVE,
        f"{path}status": ResolutionStatus.ACTIVE,
        f"{path}outcome": "automatic",
        f"{path}resolver_version": F(f"{path}artifact__resolver_version"),
        f"{path}document_entity__status": ResolutionStatus.ACTIVE,
        f"{path}document_entity__artifact__status": GraphArtifact.Status.ACTIVE,
        f"{path}collection_entity__status": ResolutionStatus.ACTIVE,
        f"{path}collection_entity__artifact_id": F(f"{path}artifact_id"),
        f"{path}manifest_input__artifact_id": F(f"{path}artifact_id"),
        f"{path}manifest_input__collection_id": F(
            f"{path}collection_entity__collection_id"
        ),
        f"{path}manifest_input__document_artifact_id": F(
            f"{path}document_entity__artifact_id"
        ),
        f"{path}manifest_input__document_id": F(f"{path}document_entity__document_id"),
    }


class CollectionEntityDocumentLinkQuerySet(CollectionArtifactChildQuerySet):
    """Expose automatic links whose complete graph path remains current."""

    def current(self):
        return self.filter(**_current_link_filters())


class CollectionEntityDocumentLink(
    CollectionArtifactChildModelMixin, ValidatedGraphModel
):
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

    objects = models.Manager.from_queryset(CollectionEntityDocumentLinkQuerySet)()

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
            if (
                self.artifact.status != GraphArtifact.Status.BUILDING
                and self._state.adding
            ):
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


def canonical_link_decision_checksum(link: object) -> str:
    """Hash one immutable canonical decision without its lifecycle status."""

    payload = {
        "canonical_entity_id": getattr(link, "canonical_entity_id", None),
        "collection_entity_id": getattr(link, "collection_entity_id", None),
        "metadata": getattr(link, "metadata", None),
        "method": getattr(link, "method", None),
        "outcome": getattr(link, "outcome", None),
        "reason": getattr(link, "reason", None),
        "resolver_version": getattr(link, "resolver_version", None),
        "score": getattr(link, "score", None),
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            {"decision_checksum": "Canonical decision is not serializable."}
        ) from exc
    return sha256(encoded).hexdigest()


class CanonicalEntityLinkQuerySet(ImmutableGraphQuerySet):
    """Expose one fail-closed current canonical assignment boundary."""

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
            outcome="automatic",
            collection_entity__status=ResolutionStatus.ACTIVE,
            collection_entity__artifact__status=GraphArtifact.Status.ACTIVE,
            canonical_entity__status=ResolutionStatus.ACTIVE,
            canonical_entity__resolver_version=resolver_version,
            canonical_entity__entity_type=F("collection_entity__entity_type"),
            canonical_entity__version_signature=F(
                "collection_entity__version_signature"
            ),
        )


class CanonicalEntityLink(ValidatedGraphModel):
    """Explicit resolver decision from a collection node to internal identity."""

    Status = ResolutionStatus

    class Outcome(models.TextChoices):
        AUTOMATIC = "automatic", "Automatic"
        CANDIDATE = "candidate", "Candidate"
        REJECTED = "rejected", "Rejected"

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
    outcome = models.CharField(max_length=16, choices=Outcome.choices)
    decision_checksum = models.CharField(max_length=64, editable=False)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.ACTIVE
    )
    reason = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    _IMMUTABLE_FIELDS = (
        "collection_entity",
        "collection_entity_id",
        "canonical_entity",
        "canonical_entity_id",
        "score",
        "method",
        "resolver_version",
        "outcome",
        "decision_checksum",
        "status",
        "reason",
        "metadata",
        "created_at",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS

    objects = models.Manager.from_queryset(CanonicalEntityLinkQuerySet)()

    @classmethod
    def _supersede_locked(
        cls, primary_keys: tuple[int, ...], *, using: str = "default"
    ) -> int:
        """Internally retire append-closed decisions after their rows are locked."""

        from django.db import connections

        if not connections[using].in_atomic_block:
            raise RuntimeError("canonical link supersession requires an atomic block")
        if type(primary_keys) is not tuple or any(
            type(value) is not int or value < 1 for value in primary_keys
        ):
            raise ValueError("canonical link primary keys are invalid")
        if primary_keys != tuple(sorted(set(primary_keys))):
            raise ValueError("canonical link primary keys must be sorted and unique")
        changed = 0
        for start in range(0, len(primary_keys), 5_000):
            query = cls.objects.using(using).filter(
                pk__in=primary_keys[start : start + 5_000],
                status__in=(
                    cls.Status.ACTIVE,
                    cls.Status.SUPPRESSED,
                    cls.Status.REJECTED,
                ),
            )
            changed += models.QuerySet.update(query, status=cls.Status.SUPERSEDED)
        return changed

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
            models.CheckConstraint(
                condition=Q(outcome__in=("automatic", "candidate", "rejected")),
                name="kg_canonical_link_outcome_valid",
            ),
            models.CheckConstraint(
                condition=(
                    Q(
                        outcome="automatic",
                        status__in=(
                            ResolutionStatus.ACTIVE,
                            ResolutionStatus.SUPERSEDED,
                        ),
                    )
                    | Q(
                        outcome="candidate",
                        status__in=(
                            ResolutionStatus.SUPPRESSED,
                            ResolutionStatus.SUPERSEDED,
                        ),
                    )
                    | Q(
                        outcome="rejected",
                        status__in=(
                            ResolutionStatus.REJECTED,
                            ResolutionStatus.SUPERSEDED,
                        ),
                    )
                ),
                name="kg_canonical_link_outcome_status",
            ),
            models.CheckConstraint(
                condition=Q(decision_checksum__regex=r"^[0-9a-f]{64}$"),
                name="kg_canonical_link_decision_hash",
            ),
            models.CheckConstraint(
                condition=~Q(resolver_version=""),
                name="kg_can_link_resolver_nonempty",
            ),
            models.CheckConstraint(
                condition=~Q(reason=""),
                name="kg_canonical_link_reason_nonempty",
            ),
            models.CheckConstraint(
                condition=(
                    Q(
                        method="embedding_similarity",
                        outcome__in=("candidate", "rejected"),
                    )
                    | ~Q(method="embedding_similarity")
                ),
                name="kg_canonical_embedding_candidate_only",
            ),
            models.CheckConstraint(
                condition=(
                    Q(
                        outcome="automatic",
                        method__in=(
                            "stable_identifier",
                            "exact_name_or_alias",
                            "defined_acronym",
                        ),
                    )
                    | Q(outcome="candidate", method="embedding_similarity")
                    | Q(
                        outcome="rejected",
                        method__in=(
                            "stable_identifier",
                            "exact_name_or_alias",
                            "defined_acronym",
                            "embedding_similarity",
                        ),
                    )
                ),
                name="kg_canonical_link_method_outcome",
            ),
            models.UniqueConstraint(
                fields=["collection_entity", "canonical_entity", "resolver_version"],
                condition=~Q(status=ResolutionStatus.SUPERSEDED),
                name="kg_collection_canonical_link_unique",
            ),
            models.UniqueConstraint(
                fields=["collection_entity", "resolver_version"],
                condition=Q(
                    outcome="automatic",
                    status=ResolutionStatus.ACTIVE,
                ),
                name="kg_one_active_canonical_target",
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
            models.Index(
                fields=["resolver_version", "status", "canonical_entity"],
                name="kg_can_link_res_can_idx",
            ),
            models.Index(
                fields=["resolver_version", "status", "collection_entity"],
                name="kg_can_link_res_src_idx",
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

    def prepare_for_persistence(self) -> None:
        if self._raw_validation_errors():
            return
        if not self.decision_checksum:
            self.decision_checksum = canonical_link_decision_checksum(self)

    def clean(self):
        super().clean()
        errors: dict[str, str] = {}
        expected_status = {
            self.Outcome.AUTOMATIC: self.Status.ACTIVE,
            self.Outcome.CANDIDATE: self.Status.SUPPRESSED,
            self.Outcome.REJECTED: self.Status.REJECTED,
        }.get(self.outcome)
        if self.status != self.Status.SUPERSEDED and self.status != expected_status:
            errors["status"] = "Status must correspond to canonical outcome."
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
        if not self.reason:
            errors["reason"] = "Canonical decisions require an audit reason."
        if type(self.metadata) is not dict:
            errors["metadata"] = "Canonical decision metadata must be an exact mapping."
        if (
            self.method == "embedding_similarity"
            and self.outcome == self.Outcome.AUTOMATIC
        ):
            errors["outcome"] = "Embedding similarity may never be automatic."
        if (
            self.outcome == self.Outcome.CANDIDATE
            and self.method != "embedding_similarity"
        ):
            errors["method"] = (
                "Only audited embedding similarity creates v1 candidates."
            )
        allowed_methods = {
            self.Outcome.AUTOMATIC: {
                "stable_identifier",
                "exact_name_or_alias",
                "defined_acronym",
            },
            self.Outcome.CANDIDATE: {"embedding_similarity"},
            self.Outcome.REJECTED: {
                "stable_identifier",
                "exact_name_or_alias",
                "defined_acronym",
                "embedding_similarity",
            },
        }
        if self.method not in allowed_methods.get(self.outcome, set()):
            errors["method"] = "Method is not valid for the canonical outcome."
        if self.collection_entity_id and self.canonical_entity_id:
            if self.collection_entity.entity_type != self.canonical_entity.entity_type:
                errors["canonical_entity"] = (
                    "Canonical endpoint type must match its source."
                )
            if self.canonical_entity.resolver_version != self.resolver_version:
                errors["resolver_version"] = (
                    "Canonical link version must match the registry identity."
                )
            if (
                self.collection_entity.version_signature
                != self.canonical_entity.version_signature
            ):
                errors["canonical_entity"] = (
                    "Canonical endpoint version must exactly match its source."
                )
        if self.decision_checksum and not re.fullmatch(
            r"[0-9a-f]{64}", self.decision_checksum
        ):
            errors["decision_checksum"] = "Decision checksum must be lowercase SHA-256."
        elif self.decision_checksum:
            try:
                expected_checksum = canonical_link_decision_checksum(self)
            except ValidationError as exc:
                errors["decision_checksum"] = str(exc)
            else:
                if self.decision_checksum != expected_checksum:
                    errors["decision_checksum"] = (
                        "Decision checksum does not match audit fields."
                    )
        if errors:
            raise ValidationError(errors)
