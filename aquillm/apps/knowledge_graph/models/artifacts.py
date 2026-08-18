from __future__ import annotations

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q


class GraphArtifact(models.Model):
    """Immutable, version-addressed output of a graph build for one scope."""

    class ScopeType(models.TextChoices):
        DOCUMENT = "document", "Document"
        COLLECTION = "collection", "Collection"

    class Status(models.TextChoices):
        BUILDING = "building", "Building"
        ACTIVE = "active", "Active"
        FAILED = "failed", "Failed"
        STALE = "stale", "Stale"
        SUPPRESSED = "suppressed", "Suppressed"
        REJECTED = "rejected", "Rejected"
        SUPERSEDED = "superseded", "Superseded"

    scope_type = models.CharField(max_length=16, choices=ScopeType.choices)
    scope_id = models.UUIDField()
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.BUILDING
    )
    source_hash = models.CharField(max_length=64)
    ontology_version = models.CharField(max_length=128)
    extractor_version = models.CharField(max_length=128)
    resolver_version = models.CharField(max_length=128)
    filter_version = models.CharField(max_length=128)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    activated_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    _IDENTITY_FIELDS = (
        "scope_type",
        "scope_id",
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_version",
    )

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["scope_type", "scope_id"],
                condition=Q(status="active"),
                name="kg_one_active_artifact_per_scope",
            ),
            models.UniqueConstraint(
                fields=[
                    "scope_type",
                    "scope_id",
                    "source_hash",
                    "ontology_version",
                    "extractor_version",
                    "resolver_version",
                    "filter_version",
                ],
                name="kg_artifact_build_identity",
            ),
        ]
        indexes = [
            models.Index(
                fields=["scope_type", "scope_id", "status"],
                name="kg_art_scope_status_idx",
            ),
            models.Index(fields=["source_hash"], name="kg_art_source_hash_idx"),
        ]

    def _validate_immutable_identity(self) -> None:
        if not self.pk:
            return
        previous = (
            type(self).objects.filter(pk=self.pk).values(*self._IDENTITY_FIELDS).first()
        )
        if previous is None:
            return
        changed = [
            field
            for field in self._IDENTITY_FIELDS
            if previous[field] != getattr(self, field)
        ]
        if changed:
            raise ValidationError(
                {
                    field: "Graph artifact build identity is immutable."
                    for field in changed
                }
            )

    def clean(self):
        super().clean()
        self._validate_immutable_identity()

    def save(self, *args, **kwargs):
        self._validate_immutable_identity()
        return super().save(*args, **kwargs)


class GraphBuildRun(models.Model):
    """Durable audit of one build attempt, independent of ephemeral evidence."""

    class Stage(models.TextChoices):
        ONTOLOGY = "ontology", "Ontology"
        EXTRACTION = "extraction", "Extraction"
        RESOLUTION = "resolution", "Resolution"
        FILTERING = "filtering", "Filtering"
        PERSISTENCE = "persistence", "Persistence"
        COMPLETE = "complete", "Complete"

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    artifact = models.ForeignKey(
        GraphArtifact,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="build_runs",
    )
    stage = models.CharField(
        max_length=16, choices=Stage.choices, default=Stage.ONTOLOGY
    )
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.PENDING
    )
    attempt = models.PositiveIntegerField(default=1)
    error_code = models.CharField(max_length=128, blank=True, default="")
    error_message = models.TextField(blank=True, default="")
    error_metadata = models.JSONField(default=dict, blank=True)
    stats = models.JSONField(default=dict, blank=True)
    timings = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(attempt__gte=1),
                name="kg_build_run_attempt_positive",
            )
        ]
        indexes = [
            models.Index(fields=["artifact", "status"], name="kg_run_art_status_idx"),
            models.Index(fields=["status", "stage"], name="kg_run_status_stage_idx"),
        ]
