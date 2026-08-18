from __future__ import annotations

import json
import re
import uuid
from hashlib import sha256

from django.core.exceptions import FieldDoesNotExist, ValidationError
from django.db import models
from django.db.models import Q

_DOCUMENT_SCOPE_PATTERN = (
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_COLLECTION_SCOPE_PATTERN = r"^[1-9][0-9]*$"
_CHECKSUM_PATTERN = r"^[0-9a-f]{64}$"


def graph_identity_checksum(namespace: object, value: object) -> str:
    """Hash a version-only identity when no richer immutable policy exists."""

    if type(namespace) is not str or not namespace or "\x00" in namespace:
        raise ValidationError("Graph identity checksum namespace is invalid.")
    if type(value) is not str or not value or "\x00" in value:
        raise ValidationError("Graph identity checksum value is invalid.")
    payload = json.dumps(
        {"namespace": namespace, "value": value},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def legacy_graph_artifact_build_key(instance: object) -> str:
    """Address pre-orchestrator artifacts without weakening their old identity."""

    fields = (
        "scope_type",
        "scope_id",
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
        "embedding_model_signature",
        "ontology_checksum",
        "filter_policy_checksum",
        "resolution_config_checksum",
        "assembly_version",
        "assembly_config_checksum",
    )
    payload = {
        "namespace": "legacy-graph-artifact-v1",
        "identity": {field: str(getattr(instance, field, "")) for field in fields},
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


ASSEMBLY_NOT_APPLICABLE_VERSION = "not-applicable"
ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM = graph_identity_checksum(
    "assembly-config", ASSEMBLY_NOT_APPLICABLE_VERSION
)


def _validate_identity_checksum(value: object, field: str) -> str:
    if type(value) is not str or not re.fullmatch(_CHECKSUM_PATTERN, value):
        raise ValidationError({field: "Graph identity must be a SHA-256 checksum."})
    return value


def _prepare_assembly_identity(instance: object) -> None:
    """Canonicalize the typed assembly identity for document/collection scope."""

    scope_type = getattr(instance, "scope_type", None)
    version = getattr(instance, "assembly_version", None)
    checksum = getattr(instance, "assembly_config_checksum", None)
    if (
        scope_type == "collection"
        and version == ASSEMBLY_NOT_APPLICABLE_VERSION
        and checksum == ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM
    ):
        # Runtime-only import keeps the persistence model independent of the
        # pure planner at module import time while giving omitted collection
        # fields the exact v1 policy address.
        from apps.knowledge_graph.graph.assembly import (
            AssemblyConfig,
            assembly_config_checksum,
        )

        config = AssemblyConfig()
        version = config.version
        checksum = assembly_config_checksum(config)
        instance.assembly_version = version
        instance.assembly_config_checksum = checksum
    if (
        type(version) is not str
        or not version
        or version != version.strip()
        or len(version) > 128
        or "\x00" in version
    ):
        raise ValidationError(
            {"assembly_version": "Assembly version must be a safe nonempty string."}
        )
    checksum = _validate_identity_checksum(checksum, "assembly_config_checksum")
    if scope_type == "document" and (
        version != ASSEMBLY_NOT_APPLICABLE_VERSION
        or checksum != ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM
    ):
        raise ValidationError(
            {
                "assembly_version": (
                    "Document artifacts require the canonical not-applicable "
                    "assembly identity."
                )
            }
        )
    if scope_type == "collection" and (
        version == ASSEMBLY_NOT_APPLICABLE_VERSION
        or checksum == ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM
    ):
        raise ValidationError(
            {
                "assembly_version": (
                    "Collection artifacts require a concrete assembly policy identity."
                )
            }
        )


def _validate_source_hash(value: object) -> str:
    if type(value) is not str or not re.fullmatch(_CHECKSUM_PATTERN, value):
        raise ValidationError(
            {"source_hash": "Graph source identity must be a SHA-256 checksum."}
        )
    return value


def canonical_graph_scope_id(scope_type: object, value: object) -> str:
    """Return a canonical string for a document UUID or positive collection PK."""

    if scope_type == "document":
        if isinstance(value, bool):
            raise ValidationError({"scope_id": "Document scope must be a UUID."})
        try:
            parsed = value if type(value) is uuid.UUID else uuid.UUID(str(value))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValidationError(
                {"scope_id": "Document scope must be a canonical UUID."}
            ) from exc
        canonical = str(parsed)
        if parsed.version is None or not re.fullmatch(
            _DOCUMENT_SCOPE_PATTERN, canonical
        ):
            raise ValidationError(
                {"scope_id": "Document scope must be a canonical RFC 4122 UUID."}
            )
        return canonical
    if scope_type == "collection":
        if isinstance(value, bool):
            raise ValidationError(
                {"scope_id": "Collection scope must be a positive decimal PK."}
            )
        if type(value) is int:
            canonical = str(value)
        elif type(value) is str:
            canonical = value
        else:
            raise ValidationError(
                {"scope_id": "Collection scope must be a positive decimal PK."}
            )
        if not re.fullmatch(_COLLECTION_SCOPE_PATTERN, canonical):
            raise ValidationError(
                {
                    "scope_id": (
                        "Collection scope must be a canonical positive decimal PK."
                    )
                }
            )
        return canonical
    raise ValidationError({"scope_type": "Graph scope type is invalid."})


def _validate_embedding_model_signature(scope_type: object, value: object) -> str:
    if type(value) is not str:
        raise ValidationError(
            {"embedding_model_signature": "Embedding signature must be a string."}
        )
    if value != value.strip() or len(value) > 512 or "\x00" in value:
        raise ValidationError(
            {
                "embedding_model_signature": (
                    "Embedding signature must be trimmed, bounded, and safe."
                )
            }
        )
    if scope_type == "document" and value:
        raise ValidationError(
            {
                "embedding_model_signature": (
                    "Document artifacts must use an empty collection "
                    "embedding signature."
                )
            }
        )
    if scope_type == "collection" and not value:
        raise ValidationError(
            {
                "embedding_model_signature": (
                    "Collection artifacts require a locked embedding signature."
                )
            }
        )
    if scope_type == "collection" and value:
        tokens = value.split(":")
        endpoint_tokens = [
            token.removeprefix("endpoint=")
            for token in tokens
            if token.startswith("endpoint=")
        ]
        if (
            len(endpoint_tokens) != 1
            or not re.fullmatch(_CHECKSUM_PATTERN, endpoint_tokens[0])
            or not {
                "dims=1024",
                "prep=kg-entity-v1",
                "max_chars=8192",
                "batch=64",
            }.issubset(tokens)
        ):
            raise ValidationError(
                {
                    "embedding_model_signature": (
                        "Collection embedding signature must lock the provider "
                        "endpoint and durable embedding contract."
                    )
                }
            )
    return value


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


class CollectionArtifactChildQuerySet(ImmutableGraphQuerySet):
    """Require collection child cleanup through the owning artifact."""

    def delete(self):
        raise ValidationError(
            {"artifact": "Graph child rows cannot be deleted directly."}
        )


class CollectionArtifactChildModelMixin:
    """Keep Task 9/10 rows append-closed outside owning-artifact deletion."""

    def delete(self, *args, **kwargs):
        raise ValidationError(
            {"artifact": "Graph child rows cannot be deleted directly."}
        )


class GraphArtifactQuerySet(ImmutableGraphQuerySet):
    """Explicit current-state boundary; shadow artifacts are opt-in only."""

    def current(self):
        return self.filter(status="active")

    def current_collection(self, collection_id: object):
        scope_id = canonical_graph_scope_id("collection", collection_id)
        return self.current().filter(scope_type="collection", scope_id=scope_id)

    def delete(self):
        if self.filter(scope_type="collection").exists():
            raise ValidationError(
                {
                    "activated_at": (
                        "Collection activation lifecycle requires dedicated locked "
                        "cleanup."
                    )
                }
            )
        document_rows = self.filter(scope_type="document")
        return super(GraphArtifactQuerySet, document_rows).delete()


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

    class OrchestrationVersion(models.IntegerChoices):
        LEGACY = 0, "Legacy"
        SCOPED_V1 = 1, "Scoped v1"

    scope_type = models.CharField(max_length=16, choices=ScopeType.choices)
    scope_id = models.CharField(max_length=64)
    build_key = models.CharField(max_length=64, default="", editable=False)
    build_generation = models.PositiveBigIntegerField(default=1, editable=False)
    orchestration_version = models.PositiveSmallIntegerField(
        choices=OrchestrationVersion.choices,
        default=OrchestrationVersion.LEGACY,
        editable=False,
    )
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.BUILDING
    )
    source_hash = models.CharField(max_length=64)
    ontology_version = models.CharField(max_length=128)
    extractor_version = models.CharField(max_length=128)
    resolver_version = models.CharField(max_length=128)
    filter_policy_version = models.CharField(max_length=128)
    embedding_model_signature = models.CharField(max_length=512, blank=True, default="")
    ontology_checksum = models.CharField(
        max_length=64, blank=True, default="", editable=False
    )
    filter_policy_checksum = models.CharField(
        max_length=64, blank=True, default="", editable=False
    )
    resolution_config_checksum = models.CharField(
        max_length=64, blank=True, default="", editable=False
    )
    assembly_version = models.CharField(
        max_length=128,
        default=ASSEMBLY_NOT_APPLICABLE_VERSION,
        editable=False,
    )
    assembly_config_checksum = models.CharField(
        max_length=64,
        default=ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM,
        editable=False,
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    activated_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    superseded_at = models.DateTimeField(null=True, blank=True)

    _IMMUTABLE_FIELDS = (
        "scope_type",
        "scope_id",
        "build_key",
        "build_generation",
        "orchestration_version",
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
        "embedding_model_signature",
        "ontology_checksum",
        "filter_policy_checksum",
        "resolution_config_checksum",
        "assembly_version",
        "assembly_config_checksum",
    )
    _QUERYSET_IMMUTABLE_FIELDS = (
        *_IMMUTABLE_FIELDS,
        "activated_at",
        "superseded_at",
    )

    objects = models.Manager.from_queryset(GraphArtifactQuerySet)()

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(scope_type__in=("document", "collection")),
                name="kg_artifact_scope_valid",
            ),
            models.CheckConstraint(
                condition=(
                    Q(scope_type="document", scope_id__regex=_DOCUMENT_SCOPE_PATTERN)
                    | Q(
                        scope_type="collection",
                        scope_id__regex=_COLLECTION_SCOPE_PATTERN,
                    )
                ),
                name="kg_artifact_typed_scope_id",
            ),
            models.CheckConstraint(
                condition=(
                    Q(scope_type="document", embedding_model_signature="")
                    | (Q(scope_type="collection") & ~Q(embedding_model_signature=""))
                ),
                name="kg_artifact_embedding_signature_scope",
            ),
            models.CheckConstraint(
                condition=(
                    Q(ontology_checksum__regex=_CHECKSUM_PATTERN)
                    & Q(filter_policy_checksum__regex=_CHECKSUM_PATTERN)
                    & Q(resolution_config_checksum__regex=_CHECKSUM_PATTERN)
                    & Q(assembly_config_checksum__regex=_CHECKSUM_PATTERN)
                ),
                name="kg_artifact_identity_checksums_valid",
            ),
            models.CheckConstraint(
                condition=(
                    Q(
                        scope_type="document",
                        assembly_version=ASSEMBLY_NOT_APPLICABLE_VERSION,
                        assembly_config_checksum=(
                            ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM
                        ),
                    )
                    | (
                        Q(scope_type="collection")
                        & ~Q(assembly_version=ASSEMBLY_NOT_APPLICABLE_VERSION)
                        & ~Q(
                            assembly_config_checksum=(
                                ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM
                            )
                        )
                    )
                ),
                name="kg_artifact_assembly_identity_scope",
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
                condition=Q(build_key__regex=_CHECKSUM_PATTERN),
                name="kg_artifact_build_key_valid",
            ),
            models.CheckConstraint(
                condition=Q(build_generation__gte=1),
                name="kg_artifact_generation_positive",
            ),
            models.CheckConstraint(
                condition=Q(orchestration_version__in=(0, 1)),
                name="kg_artifact_orchestration_version_valid",
            ),
            models.CheckConstraint(
                condition=Q(source_hash__regex=_CHECKSUM_PATTERN),
                name="kg_artifact_source_hash_valid",
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
                    "build_key",
                    "build_generation",
                ],
                name="kg_artifact_build_occurrence",
            ),
        ]
        indexes = [
            models.Index(
                fields=["scope_type", "scope_id", "status"],
                name="kg_art_scope_status_idx",
            ),
            models.Index(fields=["source_hash"], name="kg_art_source_hash_idx"),
        ]

    def prepare_for_persistence(self) -> None:
        self.scope_id = canonical_graph_scope_id(self.scope_type, self.scope_id)
        self.source_hash = _validate_source_hash(self.source_hash)
        self.embedding_model_signature = _validate_embedding_model_signature(
            self.scope_type, self.embedding_model_signature
        )
        self._prepare_identity_checksums()
        if not self.build_key:
            self.build_key = legacy_graph_artifact_build_key(self)

    def _prepare_identity_checksums(self) -> None:
        if not self.ontology_checksum:
            self.ontology_checksum = graph_identity_checksum(
                "ontology-version", self.ontology_version
            )
        if not self.filter_policy_checksum:
            self.filter_policy_checksum = graph_identity_checksum(
                "filter-policy-version", self.filter_policy_version
            )
        if not self.resolution_config_checksum:
            self.resolution_config_checksum = graph_identity_checksum(
                "resolver-version", self.resolver_version
            )
        for field in (
            "ontology_checksum",
            "filter_policy_checksum",
            "resolution_config_checksum",
        ):
            setattr(
                self, field, _validate_identity_checksum(getattr(self, field), field)
            )
        _prepare_assembly_identity(self)

    def clean(self):
        super().clean()
        if self.pk:
            previous_lifecycle = (
                type(self)
                .objects.filter(pk=self.pk)
                .values("activated_at", "superseded_at")
                .first()
            )
            if previous_lifecycle is not None:
                lifecycle_errors = {}
                for field in ("activated_at", "superseded_at"):
                    previous_value = previous_lifecycle[field]
                    if (
                        previous_value is not None
                        and getattr(self, field) != previous_value
                    ):
                        lifecycle_errors[field] = (
                            "Collection activation history is immutable once set."
                        )
                if lifecycle_errors:
                    raise ValidationError(lifecycle_errors)
        self.scope_id = canonical_graph_scope_id(self.scope_type, self.scope_id)
        self.source_hash = _validate_source_hash(self.source_hash)
        self.embedding_model_signature = _validate_embedding_model_signature(
            self.scope_type, self.embedding_model_signature
        )
        self._prepare_identity_checksums()
        if not self.build_key:
            self.build_key = legacy_graph_artifact_build_key(self)
        if not re.fullmatch(_CHECKSUM_PATTERN, self.build_key):
            raise ValidationError(
                {"build_key": "Build key must be a lowercase SHA-256 digest."}
            )

    def delete(self, *args, **kwargs):
        persisted_scope = None
        if self.pk:
            persisted_scope = (
                type(self)
                ._base_manager.filter(pk=self.pk)
                .values_list("scope_type", flat=True)
                .first()
            )
        if (persisted_scope or self.scope_type) == self.ScopeType.COLLECTION:
            raise ValidationError(
                {
                    "activated_at": (
                        "Collection activation lifecycle requires dedicated locked "
                        "cleanup."
                    )
                }
            )
        return super().delete(*args, **kwargs)


class GraphBuildRun(ValidatedGraphModel):
    """Durable audit of one build attempt, independent of ephemeral evidence."""

    class Stage(models.TextChoices):
        QUEUED = "queued", "Queued"
        EXTRACTING = "extracting", "Extracting"
        SNAPSHOTTING = "snapshotting", "Snapshotting"
        RESOLVING = "resolving", "Resolving"
        ASSEMBLING = "assembling", "Assembling"
        VALIDATING = "validating", "Validating"
        ACTIVE = "active", "Active"
        FAILED = "failed", "Failed"
        SUPERSEDED = "superseded", "Superseded"
        STALE = "stale", "Stale"

        # Typed legacy stages remain readable while Tasks 7-10 migrate to the
        # scoped orchestration lifecycle. They are not used by Task 11 runs.
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
    build_key = models.CharField(max_length=64, default="", editable=False)
    build_generation = models.PositiveBigIntegerField(default=1, editable=False)
    orchestration_version = models.PositiveSmallIntegerField(
        choices=GraphArtifact.OrchestrationVersion.choices,
        default=GraphArtifact.OrchestrationVersion.LEGACY,
        editable=False,
    )
    build_kind = models.CharField(max_length=16, choices=BuildKind.choices)
    scope_type = models.CharField(
        max_length=16, choices=GraphArtifact.ScopeType.choices
    )
    scope_id = models.CharField(max_length=64)
    source_hash = models.CharField(max_length=64)
    ontology_version = models.CharField(max_length=128)
    extractor_version = models.CharField(max_length=128)
    resolver_version = models.CharField(max_length=128)
    filter_policy_version = models.CharField(max_length=128)
    embedding_model_signature = models.CharField(max_length=512, blank=True, default="")
    ontology_checksum = models.CharField(
        max_length=64, blank=True, default="", editable=False
    )
    filter_policy_checksum = models.CharField(
        max_length=64, blank=True, default="", editable=False
    )
    resolution_config_checksum = models.CharField(
        max_length=64, blank=True, default="", editable=False
    )
    assembly_version = models.CharField(
        max_length=128,
        default=ASSEMBLY_NOT_APPLICABLE_VERSION,
        editable=False,
    )
    assembly_config_checksum = models.CharField(
        max_length=64,
        default=ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM,
        editable=False,
    )
    stage = models.CharField(
        max_length=16, choices=Stage.choices, default=Stage.ONTOLOGY
    )
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.PENDING
    )
    attempt = models.PositiveIntegerField(default=1)
    lease_owner = models.CharField(max_length=128, blank=True, default="")
    lease_generation = models.PositiveIntegerField(default=0)
    lease_expires_at = models.DateTimeField(null=True, blank=True)
    stage_marker = models.JSONField(default=dict, blank=True)
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
        "build_key",
        "build_generation",
        "orchestration_version",
        "build_kind",
        "scope_type",
        "scope_id",
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
        "embedding_model_signature",
        "ontology_checksum",
        "filter_policy_checksum",
        "resolution_config_checksum",
        "assembly_version",
        "assembly_config_checksum",
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
                condition=Q(build_kind=models.F("scope_type")),
                name="kg_build_kind_matches_scope",
            ),
            models.CheckConstraint(
                condition=(
                    Q(scope_type="document", scope_id__regex=_DOCUMENT_SCOPE_PATTERN)
                    | Q(
                        scope_type="collection",
                        scope_id__regex=_COLLECTION_SCOPE_PATTERN,
                    )
                ),
                name="kg_run_typed_scope_id",
            ),
            models.CheckConstraint(
                condition=(
                    Q(scope_type="document", embedding_model_signature="")
                    | (Q(scope_type="collection") & ~Q(embedding_model_signature=""))
                ),
                name="kg_run_embedding_signature_scope",
            ),
            models.CheckConstraint(
                condition=(
                    Q(ontology_checksum__regex=_CHECKSUM_PATTERN)
                    & Q(filter_policy_checksum__regex=_CHECKSUM_PATTERN)
                    & Q(resolution_config_checksum__regex=_CHECKSUM_PATTERN)
                    & Q(assembly_config_checksum__regex=_CHECKSUM_PATTERN)
                ),
                name="kg_run_identity_checksums_valid",
            ),
            models.CheckConstraint(
                condition=(
                    Q(
                        scope_type="document",
                        assembly_version=ASSEMBLY_NOT_APPLICABLE_VERSION,
                        assembly_config_checksum=(
                            ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM
                        ),
                    )
                    | (
                        Q(scope_type="collection")
                        & ~Q(assembly_version=ASSEMBLY_NOT_APPLICABLE_VERSION)
                        & ~Q(
                            assembly_config_checksum=(
                                ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM
                            )
                        )
                    )
                ),
                name="kg_run_assembly_identity_scope",
            ),
            models.CheckConstraint(
                condition=Q(
                    stage__in=(
                        "queued",
                        "extracting",
                        "snapshotting",
                        "resolving",
                        "assembling",
                        "validating",
                        "active",
                        "failed",
                        "superseded",
                        "stale",
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
                condition=Q(build_key__regex=_CHECKSUM_PATTERN),
                name="kg_build_key_valid",
            ),
            models.CheckConstraint(
                condition=Q(build_generation__gte=1),
                name="kg_build_generation_positive",
            ),
            models.CheckConstraint(
                condition=Q(orchestration_version__in=(0, 1)),
                name="kg_build_orchestration_version_valid",
            ),
            models.CheckConstraint(
                condition=(
                    Q(orchestration_version=0)
                    | Q(
                        orchestration_version=1,
                        build_kind="document",
                        stage__in=(
                            "queued",
                            "extracting",
                            "resolving",
                            "validating",
                            "active",
                            "failed",
                            "superseded",
                            "stale",
                        ),
                    )
                    | Q(
                        orchestration_version=1,
                        build_kind="collection",
                        stage__in=(
                            "queued",
                            "snapshotting",
                            "resolving",
                            "assembling",
                            "validating",
                            "active",
                            "failed",
                            "superseded",
                            "stale",
                        ),
                    )
                ),
                name="kg_build_stage_matches_kind",
            ),
            models.CheckConstraint(
                condition=(
                    Q(orchestration_version=0)
                    | Q(orchestration_version=1, stage="queued", status="pending")
                    | Q(
                        orchestration_version=1,
                        stage__in=(
                            "extracting",
                            "snapshotting",
                            "resolving",
                            "assembling",
                            "validating",
                        ),
                        status="running",
                    )
                    | Q(
                        orchestration_version=1,
                        stage="active",
                        status="succeeded",
                    )
                    | Q(orchestration_version=1, stage="failed", status="failed")
                    | Q(
                        orchestration_version=1,
                        stage__in=("superseded", "stale"),
                        status="cancelled",
                    )
                ),
                name="kg_build_stage_status_valid",
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
                condition=Q(source_hash__regex=_CHECKSUM_PATTERN),
                name="kg_build_source_hash_valid",
            ),
            models.CheckConstraint(
                condition=Q(attempt__gte=1),
                name="kg_build_run_attempt_positive",
            ),
            models.CheckConstraint(
                condition=Q(lease_generation__gte=0),
                name="kg_build_lease_generation_nonnegative",
            ),
            models.CheckConstraint(
                condition=(
                    Q(orchestration_version=0)
                    | Q(lease_owner="", lease_expires_at__isnull=True)
                    | (
                        Q(orchestration_version=1)
                        & ~Q(lease_owner="")
                        & Q(lease_expires_at__isnull=False)
                        & Q(lease_generation__gte=1)
                    )
                ),
                name="kg_build_lease_complete",
            ),
            models.CheckConstraint(
                condition=(
                    Q(orchestration_version=0)
                    | ~Q(stage__in=("active", "failed", "superseded", "stale"))
                    | Q(lease_owner="", lease_expires_at__isnull=True)
                ),
                name="kg_build_terminal_lease_clear",
            ),
            models.UniqueConstraint(
                fields=[
                    "build_kind",
                    "scope_type",
                    "scope_id",
                    "build_key",
                    "build_generation",
                ],
                condition=Q(orchestration_version=1),
                name="kg_build_occurrence_unique",
            ),
            models.UniqueConstraint(
                fields=["artifact"],
                condition=Q(orchestration_version=1),
                name="kg_run_artifact_occurrence_unique",
            ),
        ]
        indexes = [
            models.Index(fields=["artifact", "status"], name="kg_run_art_status_idx"),
            models.Index(fields=["status", "stage"], name="kg_run_status_stage_idx"),
            models.Index(
                fields=["build_kind", "scope_id", "build_key", "status"],
                name="kg_run_build_key_status_idx",
            ),
            models.Index(
                fields=["status", "lease_expires_at"],
                name="kg_run_status_lease_idx",
            ),
        ]

    def populate_artifact_snapshot(self) -> None:
        if not self.artifact_id:
            return
        artifact = GraphArtifact.objects.get(pk=self.artifact_id)
        self.build_kind = artifact.scope_type
        for field in (
            "build_key",
            "build_generation",
            "orchestration_version",
            "scope_type",
            "scope_id",
            "source_hash",
            "ontology_version",
            "extractor_version",
            "resolver_version",
            "filter_policy_version",
            "embedding_model_signature",
            "ontology_checksum",
            "filter_policy_checksum",
            "resolution_config_checksum",
            "assembly_version",
            "assembly_config_checksum",
        ):
            setattr(self, field, getattr(artifact, field))

    def prepare_for_persistence(self) -> None:
        if not self.pk:
            self.populate_artifact_snapshot()
        if not self.build_key and not self.artifact_id:
            # Compatibility runs created directly by Tasks 7-10 still receive
            # a durable opaque identity. Task 11 always supplies its exact,
            # reproducible build key explicitly.
            self.build_key = sha256(
                f"legacy-graph-run:{uuid.uuid4()}".encode("ascii")
            ).hexdigest()
        self.scope_id = canonical_graph_scope_id(self.scope_type, self.scope_id)
        self.source_hash = _validate_source_hash(self.source_hash)
        self.embedding_model_signature = _validate_embedding_model_signature(
            self.scope_type, self.embedding_model_signature
        )
        self._prepare_identity_checksums()

    def _prepare_identity_checksums(self) -> None:
        if not self.ontology_checksum:
            self.ontology_checksum = graph_identity_checksum(
                "ontology-version", self.ontology_version
            )
        if not self.filter_policy_checksum:
            self.filter_policy_checksum = graph_identity_checksum(
                "filter-policy-version", self.filter_policy_version
            )
        if not self.resolution_config_checksum:
            self.resolution_config_checksum = graph_identity_checksum(
                "resolver-version", self.resolver_version
            )
        for field in (
            "ontology_checksum",
            "filter_policy_checksum",
            "resolution_config_checksum",
        ):
            setattr(
                self, field, _validate_identity_checksum(getattr(self, field), field)
            )
        _prepare_assembly_identity(self)

    def clean(self):
        super().clean()
        self.scope_id = canonical_graph_scope_id(self.scope_type, self.scope_id)
        self.source_hash = _validate_source_hash(self.source_hash)
        self.embedding_model_signature = _validate_embedding_model_signature(
            self.scope_type, self.embedding_model_signature
        )
        self._prepare_identity_checksums()
        if not re.fullmatch(_CHECKSUM_PATTERN, self.build_key):
            raise ValidationError(
                {"build_key": "Build key must be a lowercase SHA-256 digest."}
            )
        if self.orchestration_version == GraphArtifact.OrchestrationVersion.SCOPED_V1:
            from apps.knowledge_graph.services.builds import (
                validate_orchestration_stage,
                validate_stage_transition,
            )

            validate_orchestration_stage(self.build_kind, self.stage, self.status)
            if self.pk:
                previous_stage = (
                    type(self)
                    .objects.filter(pk=self.pk)
                    .values_list("stage", flat=True)
                    .first()
                )
                if previous_stage is not None:
                    validate_stage_transition(
                        self.build_kind, previous_stage, self.stage
                    )
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
