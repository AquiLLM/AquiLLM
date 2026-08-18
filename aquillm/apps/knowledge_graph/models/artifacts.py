from __future__ import annotations

from django.core.exceptions import FieldDoesNotExist, ValidationError
from django.db import models
from django.db.models import Q


class ValidatedGraphQuerySet(models.QuerySet):
    """QuerySet that preserves model validation for supported bulk writes."""

    def bulk_create(self, objs, *args, **kwargs):
        objects = list(objs)
        for obj in objects:
            obj.validate_for_persistence()
        return super().bulk_create(objects, *args, **kwargs)


class ImmutableGraphQuerySet(ValidatedGraphQuerySet):
    def _reject_immutable_fields(self, fields, *, null_assignments=()) -> None:
        immutable = set(
            getattr(
                self.model,
                "_QUERYSET_IMMUTABLE_FIELDS",
                self.model._IMMUTABLE_FIELDS,
            )
        )
        changed = immutable.intersection(fields)
        nullable = set(getattr(self.model, "_NULLABLE_IMMUTABLE_UPDATE_FIELDS", ()))
        changed -= nullable.intersection(null_assignments)
        if changed:
            raise ValidationError(
                {field: "This graph field is immutable." for field in sorted(changed)}
            )

    def update(self, **kwargs):
        self._reject_immutable_fields(
            kwargs,
            null_assignments={
                field for field, value in kwargs.items() if value is None
            },
        )
        return super().update(**kwargs)

    def bulk_update(self, objs, fields, batch_size=None):
        objects = list(objs)
        null_assignments = set()
        for name in fields:
            try:
                field = self.model._meta.get_field(name)
            except FieldDoesNotExist:
                if not name.endswith("_id"):
                    raise
                field = self.model._meta.get_field(name.removesuffix("_id"))
            if all(getattr(obj, field.attname) is None for obj in objects):
                null_assignments.add(name)
        self._reject_immutable_fields(fields, null_assignments=null_assignments)
        return super().bulk_update(objects, fields, batch_size=batch_size)


class ValidatedGraphModel(models.Model):
    """Explicit validation path used by save(), create(), and bulk_create()."""

    objects = models.Manager.from_queryset(ValidatedGraphQuerySet)()
    _IMMUTABLE_FIELDS: tuple[str, ...] = ()
    _QUERYSET_IMMUTABLE_FIELDS: tuple[str, ...] = ()
    _NULLABLE_IMMUTABLE_UPDATE_FIELDS: tuple[str, ...] = ()

    class Meta:
        abstract = True

    def prepare_for_persistence(self) -> None:
        """Populate deterministic fields before full model validation."""

    def _raw_validation_errors(self) -> dict[str, str]:
        return {}

    def validate_for_persistence(self) -> None:
        """Validate one instance before any SQL is attempted."""
        self.prepare_for_persistence()
        errors = self._raw_validation_errors()
        for field in self._meta.fields:
            if not field.choices:
                continue
            value = getattr(self, field.attname)
            allowed = {choice for choice, _label in field.flatchoices}
            if value not in allowed:
                errors[field.name] = "Value is not a valid choice."
        if errors:
            raise ValidationError(errors)
        self.full_clean()

    def _validate_immutable_fields(self) -> None:
        if not self.pk or not self._IMMUTABLE_FIELDS:
            return
        comparisons: dict[str, str] = {}
        for name in self._IMMUTABLE_FIELDS:
            try:
                field = self._meta.get_field(name)
            except FieldDoesNotExist:
                if not name.endswith("_id"):
                    raise
                field = self._meta.get_field(name.removesuffix("_id"))
            lookup = field.attname if field.is_relation else field.name
            comparisons.setdefault(lookup, field.name)
        previous = type(self).objects.filter(pk=self.pk).values(*comparisons).first()
        if previous is None:
            return
        changed = {
            error_field
            for lookup, error_field in comparisons.items()
            if previous[lookup] != getattr(self, lookup)
        }
        if changed:
            raise ValidationError(
                {field: "Graph field is immutable." for field in sorted(changed)}
            )

    def clean(self):
        super().clean()
        self._validate_immutable_fields()

    def save(self, *args, **kwargs):
        self.validate_for_persistence()
        return super().save(*args, **kwargs)


class GraphArtifact(ValidatedGraphModel):
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
    filter_policy_version = models.CharField(max_length=128)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    activated_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    _IMMUTABLE_FIELDS = (
        "scope_type",
        "scope_id",
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS

    objects = models.Manager.from_queryset(ImmutableGraphQuerySet)()

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(scope_type__in=("document", "collection")),
                name="kg_artifact_scope_valid",
            ),
            models.CheckConstraint(
                condition=Q(
                    status__in=(
                        "building",
                        "active",
                        "failed",
                        "stale",
                        "suppressed",
                        "rejected",
                        "superseded",
                    )
                ),
                name="kg_artifact_status_valid",
            ),
            models.CheckConstraint(
                condition=~Q(source_hash=""),
                name="kg_artifact_source_hash_nonempty",
            ),
            models.CheckConstraint(
                condition=~Q(ontology_version=""),
                name="kg_artifact_ontology_ver_nonempty",
            ),
            models.CheckConstraint(
                condition=~Q(extractor_version=""),
                name="kg_artifact_extractor_ver_nonempty",
            ),
            models.CheckConstraint(
                condition=~Q(resolver_version=""),
                name="kg_artifact_resolver_ver_nonempty",
            ),
            models.CheckConstraint(
                condition=~Q(filter_policy_version=""),
                name="kg_artifact_filter_ver_nonempty",
            ),
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
                    "filter_policy_version",
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


class GraphBuildRun(ValidatedGraphModel):
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

    class BuildKind(models.TextChoices):
        DOCUMENT = "document", "Document"
        COLLECTION = "collection", "Collection"

    artifact = models.ForeignKey(
        GraphArtifact,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="build_runs",
    )
    build_kind = models.CharField(max_length=16, choices=BuildKind.choices)
    scope_type = models.CharField(
        max_length=16, choices=GraphArtifact.ScopeType.choices
    )
    scope_id = models.UUIDField()
    source_hash = models.CharField(max_length=64)
    ontology_version = models.CharField(max_length=128)
    extractor_version = models.CharField(max_length=128)
    resolver_version = models.CharField(max_length=128)
    filter_policy_version = models.CharField(max_length=128)
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

    _IMMUTABLE_FIELDS = (
        "artifact",
        "artifact_id",
        "build_kind",
        "scope_type",
        "scope_id",
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS
    _NULLABLE_IMMUTABLE_UPDATE_FIELDS = ("artifact", "artifact_id")

    objects = models.Manager.from_queryset(ImmutableGraphQuerySet)()

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(build_kind__in=("document", "collection")),
                name="kg_build_kind_valid",
            ),
            models.CheckConstraint(
                condition=Q(scope_type__in=GraphArtifact.ScopeType.values),
                name="kg_build_scope_valid",
            ),
            models.CheckConstraint(
                condition=Q(
                    stage__in=(
                        "ontology",
                        "extraction",
                        "resolution",
                        "filtering",
                        "persistence",
                        "complete",
                    )
                ),
                name="kg_build_stage_valid",
            ),
            models.CheckConstraint(
                condition=Q(
                    status__in=(
                        "pending",
                        "running",
                        "succeeded",
                        "failed",
                        "cancelled",
                    )
                ),
                name="kg_build_status_valid",
            ),
            models.CheckConstraint(
                condition=(
                    ~Q(source_hash="")
                    & ~Q(ontology_version="")
                    & ~Q(extractor_version="")
                    & ~Q(resolver_version="")
                    & ~Q(filter_policy_version="")
                ),
                name="kg_build_snapshot_nonempty",
            ),
            models.CheckConstraint(
                condition=Q(attempt__gte=1),
                name="kg_build_run_attempt_positive",
            ),
        ]
        indexes = [
            models.Index(fields=["artifact", "status"], name="kg_run_art_status_idx"),
            models.Index(fields=["status", "stage"], name="kg_run_status_stage_idx"),
        ]

    def populate_artifact_snapshot(self) -> None:
        if not self.artifact_id:
            return
        artifact = GraphArtifact.objects.get(pk=self.artifact_id)
        self.build_kind = artifact.scope_type
        for field in (
            "scope_type",
            "scope_id",
            "source_hash",
            "ontology_version",
            "extractor_version",
            "resolver_version",
            "filter_policy_version",
        ):
            setattr(self, field, getattr(artifact, field))

    def prepare_for_persistence(self) -> None:
        if not self.pk:
            self.populate_artifact_snapshot()

    def clean(self):
        super().clean()
        if self.artifact_id:
            artifact = GraphArtifact.objects.get(pk=self.artifact_id)
            expected = {
                field: getattr(artifact, field)
                for field in self._IMMUTABLE_FIELDS
                if field not in {"artifact", "artifact_id", "build_kind"}
            }
            expected["build_kind"] = artifact.scope_type
            mismatched = [
                field
                for field, value in expected.items()
                if getattr(self, field) != value
            ]
            if mismatched:
                raise ValidationError(
                    {
                        field: "Build snapshot must match artifact identity."
                        for field in mismatched
                    }
                )
