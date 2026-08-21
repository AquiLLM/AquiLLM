# ruff: noqa: E501
from __future__ import annotations

import re
import uuid

from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator
from django.db import models
from django.db.models import Q

from apps.documents.models import TextChunk

from .artifacts import GraphArtifact
from .projection_lifecycle_validation import validate_projection_lifecycle

_CHECKSUM = re.compile(r"^[0-9a-f]{64}$")
_MAX_BIGINT = 2**63 - 1
_FAILURE_CODES = ("source_changed", "lease_lost", "graph_unavailable", "write_failed", "validation_failed", "checksum_mismatch", "timeout", "internal_error")  # fmt: skip
# fmt: off
_EXACT_PROJECTION_CHECKSUMS = Q(graph_checksum__regex=r"^[0-9a-f]{64}$", snapshot_checksum__regex=r"^[0-9a-f]{64}$", private_mapping_checksum__regex=r"^[0-9a-f]{64}$")
_PROJECTION_LIFECYCLE = (
    Q(state="pending", collection__isnull=False, artifact__isnull=False, lease_owner="", lease_expires_at__isnull=True, failure_code="", ready_at__isnull=True, superseded_at__isnull=True)
    | (Q(state="building", collection__isnull=False, artifact__isnull=False, lease_expires_at__isnull=False, failure_code="", ready_at__isnull=True, superseded_at__isnull=True) & ~Q(lease_owner=""))
    | (Q(state="ready", collection__isnull=False, artifact__isnull=False, lease_owner="", lease_expires_at__isnull=True, failure_code="", ready_at__isnull=False, superseded_at__isnull=True) & _EXACT_PROJECTION_CHECKSUMS)
    | Q(state="failed", failure_code__in=_FAILURE_CODES, lease_owner="", lease_expires_at__isnull=True, ready_at__isnull=True, superseded_at__isnull=True)
    | (Q(state="superseded", lease_owner="", lease_expires_at__isnull=True, failure_code="", superseded_at__isnull=False) & (Q(ready_at__isnull=True) | _EXACT_PROJECTION_CHECKSUMS))
)
_OUTBOX_LIFECYCLE = Q(state="pending", published_at__isnull=True) | Q(state="published", published_at__isnull=False)
_ACTIVE_IDENTITY_FIELDS = ("collection", "artifact", "schema_version", "projection_version", "identifier_key_version", "membership_epoch")
_NONNEGATIVE_COUNTS = Q(entity_count__gte=0) & Q(relation_semantics_count__gte=0) & Q(relation_count__gte=0) & Q(evidence_count__gte=0) & Q(entity_mention_count__gte=0) & Q(chunk_count__gte=0) & Q(attempt_count__gte=0)
# fmt: on


def _validate_checksum(errors: dict[str, str], name: str, value: object) -> None:
    if type(value) is not str or _CHECKSUM.fullmatch(value) is None:
        errors[name] = "Value must be a lowercase SHA-256 checksum."


def _validate_token(errors: dict[str, str], name: str, value: object) -> None:
    # fmt: off
    if type(value) is not str or not value or value != value.strip() or any(ord(character) < 32 or ord(character) == 127 for character in value):
        errors[name] = "Value must be a bounded canonical token."
    # fmt: on


class ProjectionAuthorityModel(models.Model):
    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        self.full_clean()
        return super().save(*args, **kwargs)


class CollectionGraphMembershipState(ProjectionAuthorityModel):
    # fmt: off
    collection = models.OneToOneField("apps_collections.Collection", primary_key=True, on_delete=models.CASCADE, related_name="graph_membership_state")
    active_artifact = models.ForeignKey(GraphArtifact, null=True, blank=True, on_delete=models.SET_NULL, related_name="membership_states")
    registry_epoch = models.PositiveBigIntegerField(default=0, validators=[MaxValueValidator(_MAX_BIGINT)])
    # fmt: on
    membership_checksum = models.CharField(max_length=64)
    resolver_version = models.CharField(max_length=128)
    resolution_config_checksum = models.CharField(max_length=64)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["collection"], name="kg_membership_one_collection"
            ),  # fmt: skip
            models.CheckConstraint(
                condition=Q(registry_epoch__lte=_MAX_BIGINT),
                name="kg_membership_epoch_bounded",
            ),  # fmt: skip
        ]
        indexes = [
            models.Index(fields=["updated_at"], name="kg_membership_updated_idx")
        ]

    def clean(self) -> None:
        errors: dict[str, str] = {}
        if not 0 <= self.registry_epoch <= _MAX_BIGINT:
            errors["registry_epoch"] = "Registry epoch must fit a signed bigint."
        _validate_checksum(errors, "membership_checksum", self.membership_checksum)
        _validate_token(errors, "resolver_version", self.resolver_version)
        _validate_checksum(
            errors, "resolution_config_checksum", self.resolution_config_checksum
        )
        if errors:
            raise ValidationError(errors)


class CollectionGraphProjection(ProjectionAuthorityModel):
    class State(models.TextChoices):
        PENDING = "pending", "Pending"
        BUILDING = "building", "Building"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"
        SUPERSEDED = "superseded", "Superseded"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    generation_key = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    # fmt: off
    collection = models.ForeignKey("apps_collections.Collection", null=True, blank=True, on_delete=models.SET_NULL, related_name="graph_projections")
    collection_pk_snapshot = models.PositiveBigIntegerField(validators=[MaxValueValidator(_MAX_BIGINT)], editable=False)
    artifact = models.ForeignKey(GraphArtifact, null=True, blank=True, on_delete=models.SET_NULL, related_name="graph_projections")
    artifact_pk_snapshot = models.PositiveBigIntegerField(validators=[MaxValueValidator(_MAX_BIGINT)], editable=False)
    state = models.CharField(max_length=16, choices=State.choices, default=State.PENDING)
    schema_version = models.CharField(max_length=64)
    projection_version = models.CharField(max_length=64)
    identifier_key_version = models.CharField(max_length=64)
    membership_epoch = models.PositiveBigIntegerField(validators=[MaxValueValidator(_MAX_BIGINT)])
    membership_checksum = models.CharField(max_length=64)
    graph_checksum = models.CharField(max_length=64, blank=True, default="")
    snapshot_checksum = models.CharField(max_length=64, blank=True, default="")
    private_mapping_checksum = models.CharField(max_length=64)
    entity_count = models.PositiveIntegerField(default=0)
    relation_semantics_count = models.PositiveIntegerField(default=0)
    relation_count = models.PositiveIntegerField(default=0)
    evidence_count = models.PositiveIntegerField(default=0)
    entity_mention_count = models.PositiveIntegerField(default=0)
    chunk_count = models.PositiveIntegerField(default=0)
    attempt_count = models.PositiveSmallIntegerField(default=0)
    lease_owner = models.CharField(max_length=128, blank=True, default="")
    lease_expires_at = models.DateTimeField(null=True, blank=True)
    failure_code = models.CharField(max_length=64, choices=((value, value) for value in _FAILURE_CODES), blank=True, default="")
    # fmt: on
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    ready_at = models.DateTimeField(null=True, blank=True)
    superseded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["generation_key"], name="kg_projection_generation_unique"
            ),  # fmt: skip
            models.UniqueConstraint(
                fields=_ACTIVE_IDENTITY_FIELDS,
                condition=Q(state__in=("pending", "building", "ready")),
                name="kg_projection_active_identity_unique",
            ),
            models.CheckConstraint(
                condition=_NONNEGATIVE_COUNTS,
                name="kg_projection_nonnegative_counts",
            ),
            models.CheckConstraint(
                condition=(
                    Q(lease_owner="", lease_expires_at__isnull=True)
                    | (~Q(lease_owner="") & Q(lease_expires_at__isnull=False))
                ),
                name="kg_projection_lease_pair",
            ),
            models.CheckConstraint(
                condition=_PROJECTION_LIFECYCLE,
                name="kg_projection_lifecycle_valid",
            ),
        ]
        # fmt: off
        indexes = [
            models.Index(fields=["state", "updated_at", "id"], name="kg_proj_state_updated_idx"),
            models.Index(fields=["state", "lease_expires_at", "id"], name="kg_projection_lease_idx"),
        ]
        # fmt: on

    def clean(self) -> None:
        errors: dict[str, str] = {}
        for name in ("collection_pk_snapshot", "artifact_pk_snapshot"):
            if not 1 <= getattr(self, name) <= _MAX_BIGINT:
                errors[name] = "Tombstone snapshot must fit a positive signed bigint."
        if not 0 <= self.membership_epoch <= _MAX_BIGINT:
            errors["membership_epoch"] = "Membership epoch must fit a signed bigint."
        for name in ("schema_version", "projection_version", "identifier_key_version"):
            _validate_token(errors, name, getattr(self, name))
        _validate_checksum(errors, "membership_checksum", self.membership_checksum)
        _validate_checksum(
            errors, "private_mapping_checksum", self.private_mapping_checksum
        )
        for name in ("graph_checksum", "snapshot_checksum"):
            if getattr(self, name):
                _validate_checksum(errors, name, getattr(self, name))
        for name in (
            "entity_count",
            "relation_semantics_count",
            "relation_count",
            "evidence_count",
            "entity_mention_count",
            "chunk_count",
        ):
            if getattr(self, name) < 0:
                errors[name] = "Projection counts must be nonnegative."
        validate_projection_lifecycle(self, errors, _FAILURE_CODES)
        if errors:
            raise ValidationError(errors)


class ProjectionChunkReference(ProjectionAuthorityModel):
    # fmt: off
    projection = models.ForeignKey(CollectionGraphProjection, on_delete=models.CASCADE, related_name="chunk_references")
    projection_chunk_key = models.CharField(max_length=64)
    chunk = models.ForeignKey(TextChunk, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    integer_chunk_pk = models.PositiveBigIntegerField(validators=[MaxValueValidator(_MAX_BIGINT)], editable=False)
    # fmt: on
    document_uuid = models.UUIDField(editable=False)
    chunk_number = models.PositiveIntegerField(editable=False)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["projection", "projection_chunk_key"],
                name="kg_projection_chunk_key_unique",
            ),  # fmt: skip
            models.UniqueConstraint(
                fields=["projection", "document_uuid", "chunk_number"],
                name="kg_projection_chunk_coordinate_unique",
            ),  # fmt: skip
        ]

    def clean(self) -> None:
        errors: dict[str, str] = {}
        _validate_checksum(errors, "projection_chunk_key", self.projection_chunk_key)
        if not 1 <= self.integer_chunk_pk <= _MAX_BIGINT:
            errors["integer_chunk_pk"] = (
                "Chunk tombstone must fit a positive signed bigint."
            )
        if errors:
            raise ValidationError(errors)


class GraphProjectionOutbox(ProjectionAuthorityModel):
    class Operation(models.TextChoices):
        PROJECT = "project", "Project"
        PRUNE = "prune", "Prune"

    class State(models.TextChoices):
        PENDING = "pending", "Pending"
        PUBLISHED = "published", "Published"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    projection = models.ForeignKey(CollectionGraphProjection, on_delete=models.CASCADE, related_name="outbox_entries")  # fmt: skip
    operation = models.CharField(max_length=16, choices=Operation.choices)
    state = models.CharField(
        max_length=16, choices=State.choices, default=State.PENDING
    )
    attempt_count = models.PositiveSmallIntegerField(default=0)
    next_attempt_at = models.DateTimeField()
    published_at = models.DateTimeField(null=True, blank=True)
    last_failure_code = models.CharField(max_length=64, blank=True, default="")

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["projection", "operation"],
                name="kg_projection_outbox_operation_unique",
            ),  # fmt: skip
            models.CheckConstraint(
                condition=_OUTBOX_LIFECYCLE,
                name="kg_projection_outbox_state_valid",
            ),
        ]
        indexes = [models.Index(fields=["state", "next_attempt_at", "id"], name="kg_projection_outbox_due_idx")]  # fmt: skip

    def clean(self) -> None:
        errors: dict[str, str] = {}
        if self.state == self.State.PUBLISHED and self.published_at is None:
            errors["published_at"] = "Published outbox rows require published_at."
        if self.state == self.State.PENDING and self.published_at is not None:
            errors["published_at"] = "Pending outbox rows cannot be published."
        if self.last_failure_code and (
            len(self.last_failure_code) > 64
            or re.fullmatch(r"[a-z][a-z0-9_]*", self.last_failure_code) is None
        ):
            errors["last_failure_code"] = "Failure code must be a fixed safe token."
        if errors:
            raise ValidationError(errors)
