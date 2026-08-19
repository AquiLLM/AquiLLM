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
_ERROR_CODE_PATTERN = r"^[a-z][a-z0-9_]{0,127}$"
_RESNAPSHOT_RECONCILING_ERROR_CODES = frozenset(
    {"resnapshot_pending", "resnapshot_churn"}
)
_RESNAPSHOT_FINAL_ERROR_CODES = frozenset(
    {"resnapshot_churn", "scope_deleted", "scope_ineligible"}
)


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


def _activation_audit_values(artifact: object, run: object) -> dict[str, object]:
    """Freeze one exact terminal occurrence before its graph rows may be removed."""

    artifact_fields = (
        "pk",
        "rebuild_request_id",
        "evaluation_only",
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
        "status",
        "activated_at",
        "completed_at",
        "superseded_at",
    )
    run_fields = (
        "pk",
        "artifact_id",
        "rebuild_request_id",
        "evaluation_only",
        "build_kind",
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
        "status",
        "stage",
        "stage_marker",
        "started_at",
        "finished_at",
    )
    payload = {
        "namespace": "graph-rebuild-activation-audit-v1",
        "artifact": {
            field: getattr(artifact, field, None) for field in artifact_fields
        },
        "run": {field: getattr(run, field, None) for field in run_fields},
    }
    signature = sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "activated_artifact_pk": getattr(artifact, "pk", None),
        "activated_run_pk": getattr(run, "pk", None),
        "activated_build_key": getattr(artifact, "build_key", ""),
        "activated_build_generation": getattr(
            artifact,
            "build_generation",
            None,
        ),
        "activated_source_hash": getattr(artifact, "source_hash", ""),
        "activated_occurrence_signature": signature,
    }


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
        if int(canonical) > 2**63 - 1:
            raise ValidationError(
                {"scope_id": "Collection scope must fit a signed bigint PK."}
            )
        return canonical
    raise ValidationError({"scope_type": "Graph scope type is invalid."})


def _cached_or_persisted_rebuild_request(instance: object):
    request_id = getattr(instance, "rebuild_request_id", None)
    evaluation_only = getattr(instance, "evaluation_only", None)
    if request_id is None:
        if evaluation_only is True:
            raise ValidationError(
                {"rebuild_request": "Evaluation occurrences require a request."}
            )
        return None
    cached = getattr(instance, "_state", None)
    fields_cache = getattr(cached, "fields_cache", {})
    request = fields_cache.get("rebuild_request")
    if request is None:
        request = GraphRebuildRequest.objects.filter(pk=request_id).first()
    if request is None:
        raise ValidationError({"rebuild_request": "Rebuild request does not exist."})
    return request


def _validate_rebuild_occurrence_correlation(instance: object) -> None:
    """Validate the cross-row request marker and exact immutable scope snapshot."""

    request = _cached_or_persisted_rebuild_request(instance)
    if request is None:
        return
    if request.evaluation_only is not getattr(instance, "evaluation_only", None):
        raise ValidationError(
            {"evaluation_only": "Occurrence marker must match its rebuild request."}
        )
    scope_type = getattr(instance, "scope_type", None)
    scope_id = getattr(instance, "scope_id", None)
    source_hash = getattr(instance, "source_hash", None)
    if request.scope_type == GraphRebuildRequest.ScopeType.ALL:
        raise ValidationError(
            {"rebuild_request": "Operator parent requests cannot own occurrences."}
        )
    if scope_type == GraphArtifact.ScopeType.COLLECTION:
        if (
            request.scope_type != GraphRebuildRequest.ScopeType.COLLECTION
            or request.scope_id != scope_id
            or request.expected_aggregate_signature != source_hash
        ):
            raise ValidationError(
                {"rebuild_request": "Collection occurrence is outside its request."}
            )
        return
    if scope_type != GraphArtifact.ScopeType.DOCUMENT:
        raise ValidationError({"scope_type": "Graph occurrence scope is invalid."})
    matching = tuple(
        row
        for row in request.requested_documents
        if row.get("document_id") == scope_id and row.get("source_hash") == source_hash
    )
    if len(matching) != 1:
        raise ValidationError(
            {"rebuild_request": "Document occurrence is outside its request snapshot."}
        )
    if request.scope_type == GraphRebuildRequest.ScopeType.DOCUMENT:
        valid_scope = request.scope_id == scope_id
    else:
        valid_scope = (
            request.scope_type == GraphRebuildRequest.ScopeType.COLLECTION
            and matching[0].get("collection_id") == int(request.scope_id)
        )
    if not valid_scope:
        raise ValidationError(
            {"rebuild_request": "Document occurrence is outside its request scope."}
        )


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


class GraphRebuildRequestQuerySet(ImmutableGraphQuerySet):
    """Append-retain durable operator request and lineage audit rows."""

    def delete(self):
        raise ValidationError(
            {"id": "Graph rebuild request audit rows cannot be deleted."}
        )

    def _terminalize_operator_parent(
        self,
        *,
        request_id: uuid.UUID,
        status: str,
        successes: int,
        failures: int,
        expected_children: int,
        error_code: str,
        completed_at: object,
    ) -> int:
        """Apply one exact set-wise ALL aggregation under the caller's row lock."""

        expected_status = (
            "succeeded" if failures == 0 else "failed" if successes == 0 else "partial"
        )
        if (
            type(request_id) is not uuid.UUID
            or type(successes) is not int
            or type(failures) is not int
            or type(expected_children) is not int
            or min(successes, failures, expected_children) < 0
            or successes + failures != expected_children
            or status != expected_status
            or error_code != ("" if failures == 0 else "child_rebuild_failed")
            or completed_at is None
        ):
            raise ValidationError(
                {"status": "Operator terminal aggregation is not exact."}
            )
        queryset = self.filter(
            pk=request_id,
            scope_type="all",
            status="running",
            enumeration_complete=True,
            expected_child_count=expected_children,
        )
        return models.QuerySet.update(
            queryset,
            status=status,
            completed_collection_count=successes,
            failed_collection_count=failures,
            error_code=error_code,
            completed_at=completed_at,
            updated_at=completed_at,
        )


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
    collection_scope = models.ForeignKey(
        "apps_collections.Collection",
        null=True,
        blank=True,
        on_delete=models.DO_NOTHING,
        related_name="knowledge_graph_artifacts",
        editable=False,
    )
    rebuild_request = models.ForeignKey(
        "apps_knowledge_graph.GraphRebuildRequest",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="artifacts",
        editable=False,
    )
    evaluation_only = models.BooleanField(default=False, editable=False)
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
        "collection_scope",
        "collection_scope_id",
        "rebuild_request",
        "rebuild_request_id",
        "evaluation_only",
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
                    Q(scope_type="document", collection_scope__isnull=True)
                    | Q(scope_type="collection", collection_scope__isnull=False)
                ),
                name="kg_artifact_collection_scope_xor",
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
            models.CheckConstraint(
                condition=(
                    Q(evaluation_only=False)
                    | (Q(rebuild_request__isnull=False) & ~Q(status="active"))
                ),
                name="kg_artifact_eval_noncurrent",
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
            models.UniqueConstraint(
                fields=["scope_type", "scope_id", "build_generation"],
                condition=Q(orchestration_version=1),
                name="kg_artifact_scope_generation_unique",
            ),
            models.UniqueConstraint(
                fields=["rebuild_request", "scope_type", "scope_id"],
                condition=Q(rebuild_request__isnull=False),
                name="kg_artifact_request_scope_unique",
            ),
        ]
        indexes = [
            models.Index(
                fields=["scope_type", "scope_id", "status"],
                name="kg_art_scope_status_idx",
            ),
            models.Index(fields=["source_hash"], name="kg_art_source_hash_idx"),
            models.Index(
                fields=["status", "completed_at", "id"],
                name="kg_art_terminal_idx",
            ),
            models.Index(
                fields=["status", "superseded_at", "id"],
                name="kg_art_superseded_idx",
            ),
        ]

    def prepare_for_persistence(self) -> None:
        self.scope_id = canonical_graph_scope_id(self.scope_type, self.scope_id)
        self._validate_collection_scope()
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

    def _validate_collection_scope(self) -> None:
        if self.scope_type == self.ScopeType.DOCUMENT:
            if self.collection_scope_id is not None:
                raise ValidationError(
                    {
                        "collection_scope": (
                            "Document artifacts cannot own a collection scope."
                        )
                    }
                )
            return
        if self.scope_type == self.ScopeType.COLLECTION:
            if self.collection_scope_id is None:
                self.collection_scope_id = int(self.scope_id)
            if str(self.collection_scope_id) != self.scope_id:
                raise ValidationError(
                    {
                        "collection_scope": (
                            "Collection scope must match the canonical scope ID."
                        )
                    }
                )

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
        self._validate_collection_scope()
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
        if self.evaluation_only and (
            self.rebuild_request_id is None or self.status == self.Status.ACTIVE
        ):
            raise ValidationError(
                {"evaluation_only": "Evaluation artifacts cannot become current."}
            )
        _validate_rebuild_occurrence_correlation(self)

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
    rebuild_request = models.ForeignKey(
        "apps_knowledge_graph.GraphRebuildRequest",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="build_runs",
        editable=False,
    )
    evaluation_only = models.BooleanField(default=False, editable=False)
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
        "rebuild_request",
        "rebuild_request_id",
        "evaluation_only",
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
            models.CheckConstraint(
                condition=(
                    Q(evaluation_only=False)
                    | (
                        Q(rebuild_request__isnull=False)
                        & ~Q(stage="active")
                        & ~Q(status="succeeded")
                    )
                ),
                name="kg_build_eval_noncurrent",
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
                fields=[
                    "build_kind",
                    "scope_type",
                    "scope_id",
                    "build_generation",
                ],
                condition=Q(orchestration_version=1),
                name="kg_run_scope_generation_unique",
            ),
            models.UniqueConstraint(
                fields=["artifact"],
                condition=Q(orchestration_version=1),
                name="kg_run_artifact_occurrence_unique",
            ),
            models.UniqueConstraint(
                fields=["rebuild_request", "scope_type", "scope_id"],
                condition=Q(rebuild_request__isnull=False),
                name="kg_run_request_scope_unique",
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
            models.Index(
                fields=["status", "stage", "finished_at", "id"],
                name="kg_run_terminal_idx",
            ),
            models.Index(
                fields=[
                    "build_kind",
                    "scope_type",
                    "scope_id",
                    "build_generation",
                ],
                name="kg_run_scope_gen_idx",
            ),
        ]

    def populate_artifact_snapshot(self) -> None:
        if not self.artifact_id:
            return
        artifact = GraphArtifact.objects.get(pk=self.artifact_id)
        self.build_kind = artifact.scope_type
        for field in (
            "rebuild_request_id",
            "evaluation_only",
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
        if self.evaluation_only and (
            self.rebuild_request_id is None
            or self.stage == self.Stage.ACTIVE
            or self.status == self.Status.SUCCEEDED
        ):
            raise ValidationError(
                {"evaluation_only": "Evaluation runs cannot become current."}
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
        _validate_rebuild_occurrence_correlation(self)


class GraphRebuildRequest(ValidatedGraphModel):
    """Durable operator request spanning forced document and collection builds."""

    class ScopeType(models.TextChoices):
        DOCUMENT = "document", "Document"
        COLLECTION = "collection", "Collection"
        ALL = "all", "All"

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"
        PARTIAL = "partial", "Partial"

    class PublicationState(models.TextChoices):
        NOT_APPLICABLE = "not_applicable", "Not applicable"
        PENDING = "pending", "Pending"
        PUBLISHED = "published", "Published"
        FAILED = "failed", "Failed"

    _ACTIVATION_AUDIT_FIELDS = (
        "activated_artifact_pk",
        "activated_run_pk",
        "activated_build_key",
        "activated_build_generation",
        "activated_source_hash",
        "activated_occurrence_signature",
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    parent_request = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="child_requests",
        editable=False,
    )
    predecessor_request = models.OneToOneField(
        "self",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="successor_request",
        editable=False,
    )
    lineage_root = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="replacement_requests",
        editable=False,
    )
    scope_type = models.CharField(max_length=16, choices=ScopeType.choices)
    scope_id = models.CharField(max_length=64, blank=True, default="")
    requested_documents = models.JSONField(default=list, blank=True)
    expected_aggregate_signature = models.CharField(
        max_length=64, blank=True, default="", editable=False
    )
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.QUEUED
    )
    evaluation_only = models.BooleanField(default=False, editable=False)
    document_count = models.PositiveIntegerField(default=0, editable=False)
    completed_document_count = models.PositiveIntegerField(default=0)
    terminal_failure_count = models.PositiveIntegerField(default=0)
    collection_count = models.PositiveIntegerField(default=0)
    completed_collection_count = models.PositiveIntegerField(default=0)
    failed_collection_count = models.PositiveIntegerField(default=0)
    enumeration_high_water = models.PositiveBigIntegerField(null=True, blank=True)
    enumeration_cursor = models.PositiveBigIntegerField(default=0)
    enumeration_complete = models.BooleanField(default=False)
    expected_child_count = models.PositiveIntegerField(default=0)
    document_publish_cursor = models.PositiveIntegerField(default=0)
    document_publication_state = models.CharField(
        max_length=16,
        choices=PublicationState.choices,
        default=PublicationState.PENDING,
    )
    collection_build_key = models.CharField(
        max_length=64, blank=True, default="", editable=False
    )
    collection_publication_state = models.CharField(
        max_length=16,
        choices=PublicationState.choices,
        default=PublicationState.NOT_APPLICABLE,
    )
    activated_artifact_pk = models.PositiveBigIntegerField(
        null=True,
        blank=True,
        editable=False,
    )
    activated_run_pk = models.PositiveBigIntegerField(
        null=True,
        blank=True,
        editable=False,
    )
    activated_build_key = models.CharField(
        max_length=64,
        blank=True,
        default="",
        editable=False,
    )
    activated_build_generation = models.PositiveBigIntegerField(
        null=True,
        blank=True,
        editable=False,
    )
    activated_source_hash = models.CharField(
        max_length=64,
        blank=True,
        default="",
        editable=False,
    )
    activated_occurrence_signature = models.CharField(
        max_length=64,
        blank=True,
        default="",
        editable=False,
    )
    error_code = models.CharField(max_length=128, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    collection_refresh_enqueued_at = models.DateTimeField(null=True, blank=True)
    collection_refresh_published_at = models.DateTimeField(null=True, blank=True)

    _IMMUTABLE_FIELDS = (
        "id",
        "parent_request",
        "parent_request_id",
        "predecessor_request",
        "predecessor_request_id",
        "lineage_root",
        "lineage_root_id",
        "scope_type",
        "scope_id",
        "requested_documents",
        "evaluation_only",
        "document_count",
        "enumeration_high_water",
        "created_at",
    )
    _QUERYSET_IMMUTABLE_FIELDS = (
        *_IMMUTABLE_FIELDS,
        "expected_aggregate_signature",
        "status",
        "completed_document_count",
        "terminal_failure_count",
        "collection_count",
        "completed_collection_count",
        "failed_collection_count",
        "enumeration_cursor",
        "enumeration_complete",
        "expected_child_count",
        "document_publish_cursor",
        "document_publication_state",
        "collection_build_key",
        "collection_publication_state",
        "activated_artifact_pk",
        "activated_run_pk",
        "activated_build_key",
        "activated_build_generation",
        "activated_source_hash",
        "activated_occurrence_signature",
        "error_code",
        "started_at",
        "completed_at",
        "collection_refresh_enqueued_at",
        "collection_refresh_published_at",
    )

    objects = models.Manager.from_queryset(GraphRebuildRequestQuerySet)()

    class Meta:
        app_label = "apps_knowledge_graph"
        indexes = [
            models.Index(fields=["status", "created_at"], name="kg_rebuild_status_idx"),
            models.Index(
                fields=["scope_type", "scope_id", "created_at"],
                name="kg_rebuild_scope_idx",
            ),
        ]
        constraints = [
            models.CheckConstraint(
                condition=Q(scope_type__in=("document", "collection", "all")),
                name="kg_rebuild_scope_valid",
            ),
            models.CheckConstraint(
                condition=(
                    Q(scope_type="all", scope_id="")
                    | Q(scope_type="document", scope_id__regex=_DOCUMENT_SCOPE_PATTERN)
                    | Q(
                        scope_type="collection",
                        scope_id__regex=_COLLECTION_SCOPE_PATTERN,
                    )
                ),
                name="kg_rebuild_typed_scope",
            ),
            models.CheckConstraint(
                condition=Q(expected_aggregate_signature="")
                | Q(expected_aggregate_signature__regex=_CHECKSUM_PATTERN),
                name="kg_rebuild_expected_hash",
            ),
            models.CheckConstraint(
                condition=Q(completed_document_count__lte=models.F("document_count")),
                name="kg_rebuild_completed_bounded",
            ),
            models.CheckConstraint(
                condition=Q(terminal_failure_count__lte=models.F("document_count")),
                name="kg_rebuild_failures_bounded",
            ),
            models.CheckConstraint(
                condition=Q(
                    completed_document_count__lte=(
                        models.F("document_count") - models.F("terminal_failure_count")
                    )
                ),
                name="kg_rebuild_outcomes_bounded",
            ),
            models.CheckConstraint(
                condition=Q(
                    completed_collection_count__lte=models.F("collection_count")
                ),
                name="kg_rebuild_collections_completed_bounded",
            ),
            models.CheckConstraint(
                condition=Q(failed_collection_count__lte=models.F("collection_count")),
                name="kg_rebuild_collections_failed_bounded",
            ),
            models.CheckConstraint(
                condition=Q(
                    completed_collection_count__lte=(
                        models.F("collection_count")
                        - models.F("failed_collection_count")
                    )
                ),
                name="kg_rebuild_collection_outcomes_bounded",
            ),
            models.CheckConstraint(
                condition=Q(document_publish_cursor__lte=models.F("document_count")),
                name="kg_rebuild_publish_cursor_bounded",
            ),
            models.CheckConstraint(
                condition=Q(enumeration_cursor__lte=models.F("enumeration_high_water"))
                | Q(enumeration_high_water__isnull=True, enumeration_cursor=0),
                name="kg_rebuild_enumeration_cursor_bounded",
            ),
            models.CheckConstraint(
                condition=Q(
                    status__in=(
                        "queued",
                        "running",
                        "succeeded",
                        "failed",
                        "partial",
                    )
                ),
                name="kg_rebuild_status_valid",
            ),
            models.CheckConstraint(
                condition=Q(evaluation_only=False) | Q(scope_type="collection"),
                name="kg_rebuild_eval_collection",
            ),
            models.CheckConstraint(
                condition=Q(scope_type="collection")
                | Q(expected_aggregate_signature=""),
                name="kg_rebuild_expected_scope",
            ),
            models.CheckConstraint(
                condition=Q(collection_refresh_enqueued_at__isnull=True)
                | Q(scope_type="collection"),
                name="kg_rebuild_refresh_scope",
            ),
            models.CheckConstraint(
                condition=(
                    Q(
                        status__in=("succeeded", "failed", "partial"),
                        completed_at__isnull=False,
                    )
                    | Q(
                        status__in=("queued", "running"),
                        completed_at__isnull=True,
                    )
                ),
                name="kg_rebuild_terminal_time",
            ),
            models.CheckConstraint(
                condition=(
                    Q(status="succeeded")
                    | Q(
                        activated_artifact_pk__isnull=True,
                        activated_run_pk__isnull=True,
                        activated_build_key="",
                        activated_build_generation__isnull=True,
                        activated_source_hash="",
                        activated_occurrence_signature="",
                    )
                ),
                name="kg_rebuild_activation_terminal",
            ),
            models.CheckConstraint(
                condition=(
                    Q(scope_type="all")
                    | ~Q(status="succeeded")
                    | Q(
                        activated_artifact_pk__isnull=False,
                        activated_run_pk__isnull=False,
                        activated_build_key__regex=_CHECKSUM_PATTERN,
                        activated_build_generation__isnull=False,
                        activated_source_hash__regex=_CHECKSUM_PATTERN,
                        activated_occurrence_signature__regex=_CHECKSUM_PATTERN,
                    )
                ),
                name="kg_rebuild_scoped_success_artifact",
            ),
            models.CheckConstraint(
                condition=Q(error_code="") | Q(error_code__regex=_ERROR_CODE_PATTERN),
                name="kg_rebuild_error_code_safe",
            ),
        ]

    def prepare_for_persistence(self) -> None:
        if self.scope_type == self.ScopeType.ALL:
            self.scope_id = ""
        else:
            self.scope_id = canonical_graph_scope_id(self.scope_type, self.scope_id)
        if type(self.requested_documents) is not list:
            raise ValidationError(
                {
                    "requested_documents": (
                        "Requested documents must be a bounded JSON list."
                    )
                }
            )
        if len(self.requested_documents) > 10_000:
            raise ValidationError(
                {"requested_documents": "Requested document snapshot exceeds its cap."}
            )
        required = {
            "document_id",
            "document_pkid",
            "model_label",
            "collection_id",
            "source_hash",
        }
        from apps.documents.models import DESCENDED_FROM_DOCUMENT

        allowed_model_labels = {
            model._meta.label_lower for model in DESCENDED_FROM_DOCUMENT
        }
        canonical_rows = []
        for row in self.requested_documents:
            if type(row) is not dict or set(row) != required:
                raise ValidationError(
                    {
                        "requested_documents": (
                            "Requested documents must contain exact scalar snapshots."
                        )
                    }
                )
            try:
                document_id = str(uuid.UUID(row["document_id"]))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValidationError(
                    {"requested_documents": "Document snapshot UUID is invalid."}
                ) from exc
            if document_id != row["document_id"]:
                raise ValidationError(
                    {"requested_documents": "Document snapshot UUID is not canonical."}
                )
            if (
                type(row["document_pkid"]) is not int
                or row["document_pkid"] <= 0
                or type(row["collection_id"]) is not int
                or row["collection_id"] <= 0
                or type(row["model_label"]) is not str
                or row["model_label"] not in allowed_model_labels
                or type(row["source_hash"]) is not str
                or not re.fullmatch(_CHECKSUM_PATTERN, row["source_hash"])
            ):
                raise ValidationError(
                    {"requested_documents": "Document snapshot scalar is invalid."}
                )
            canonical_rows.append(
                (
                    row["model_label"],
                    row["document_id"],
                    row["document_pkid"],
                )
            )
        if canonical_rows != sorted(canonical_rows) or len(canonical_rows) != len(
            set(row[1] for row in canonical_rows)
        ):
            raise ValidationError(
                {
                    "requested_documents": (
                        "Requested document snapshots must be ordered and UUID-unique."
                    )
                }
            )
        if self.document_count != len(self.requested_documents):
            raise ValidationError(
                {"document_count": "Document count must match the request snapshot."}
            )
        if self.scope_type == self.ScopeType.ALL and self.requested_documents:
            raise ValidationError(
                {"requested_documents": "Operator parent requests use child snapshots."}
            )
        if self.scope_type == self.ScopeType.DOCUMENT and self.document_count != 1:
            raise ValidationError(
                {"document_count": "Document rebuild requests require one snapshot."}
            )
        if self.scope_type == self.ScopeType.DOCUMENT and self.collection_count != 0:
            raise ValidationError(
                {"collection_count": "Document requests have no collection unit."}
            )
        if self.scope_type == self.ScopeType.COLLECTION and self.collection_count != 1:
            raise ValidationError(
                {"collection_count": "Collection requests require one collection unit."}
            )
        if self.scope_type == self.ScopeType.ALL:
            if (
                type(self.enumeration_high_water) is not int
                or self.enumeration_high_water < 0
                or self.collection_count != self.expected_child_count
            ):
                raise ValidationError(
                    {"enumeration_high_water": "Operator enumeration state is invalid."}
                )
        elif (
            self.enumeration_high_water is not None
            or self.enumeration_cursor != 0
            or self.enumeration_complete
            or self.expected_child_count != 0
        ):
            raise ValidationError(
                {"enumeration_cursor": "Scoped requests cannot enumerate children."}
            )
        if self.scope_type == self.ScopeType.DOCUMENT and (
            self.requested_documents[0]["document_id"] != self.scope_id
        ):
            raise ValidationError(
                {"requested_documents": "Document snapshot must match request scope."}
            )
        if self.scope_type == self.ScopeType.COLLECTION and any(
            row["collection_id"] != int(self.scope_id)
            for row in self.requested_documents
        ):
            raise ValidationError(
                {
                    "requested_documents": (
                        "Document snapshots must match collection scope."
                    )
                }
            )
        if self.expected_aggregate_signature:
            _validate_identity_checksum(
                self.expected_aggregate_signature, "expected_aggregate_signature"
            )
        if self.error_code and not re.fullmatch(_ERROR_CODE_PATTERN, self.error_code):
            raise ValidationError(
                {"error_code": "Error code must be a bounded private identifier."}
            )
        if self.collection_build_key:
            _validate_identity_checksum(
                self.collection_build_key,
                "collection_build_key",
            )
        if (
            self.completed_document_count + self.terminal_failure_count
            > self.document_count
        ):
            raise ValidationError(
                {"completed_document_count": "Document outcomes exceed the snapshot."}
            )
        if (
            self.completed_collection_count + self.failed_collection_count
            > self.collection_count
        ):
            raise ValidationError(
                {"completed_collection_count": "Collection outcomes exceed scope."}
            )
        if self.document_publish_cursor > self.document_count:
            raise ValidationError(
                {"document_publish_cursor": "Publication cursor exceeds snapshot."}
            )
        self._validate_lineage_shape()

    def clean(self):
        super().clean()
        self._validate_success_activation()
        if not self.pk:
            return
        previous = (
            type(self)
            .objects.filter(pk=self.pk)
            .values(
                "status",
                "expected_aggregate_signature",
                "completed_document_count",
                "terminal_failure_count",
                "collection_count",
                "completed_collection_count",
                "failed_collection_count",
                "enumeration_cursor",
                "enumeration_complete",
                "expected_child_count",
                "document_publish_cursor",
                "document_publication_state",
                "collection_build_key",
                "collection_publication_state",
                *self._ACTIVATION_AUDIT_FIELDS,
                "error_code",
                "started_at",
                "completed_at",
                "collection_refresh_enqueued_at",
                "collection_refresh_published_at",
            )
            .first()
        )
        if previous is None:
            return
        allowed = {
            self.Status.QUEUED: {
                self.Status.QUEUED,
                self.Status.RUNNING,
                self.Status.FAILED,
            },
            self.Status.RUNNING: {
                self.Status.RUNNING,
                self.Status.SUCCEEDED,
                self.Status.FAILED,
                self.Status.PARTIAL,
            },
            self.Status.SUCCEEDED: {self.Status.SUCCEEDED},
            self.Status.FAILED: {self.Status.FAILED},
            self.Status.PARTIAL: {self.Status.PARTIAL},
        }
        if self.status not in allowed.get(previous["status"], set()):
            raise ValidationError({"status": "Rebuild request transition is invalid."})
        terminal = {
            self.Status.SUCCEEDED,
            self.Status.FAILED,
            self.Status.PARTIAL,
        }
        mutable_fields = (
            "expected_aggregate_signature",
            "completed_document_count",
            "terminal_failure_count",
            "collection_count",
            "completed_collection_count",
            "failed_collection_count",
            "enumeration_cursor",
            "enumeration_complete",
            "expected_child_count",
            "document_publish_cursor",
            "document_publication_state",
            "collection_build_key",
            "collection_publication_state",
            *self._ACTIVATION_AUDIT_FIELDS,
            "error_code",
            "started_at",
            "completed_at",
            "collection_refresh_enqueued_at",
            "collection_refresh_published_at",
        )
        if previous["status"] in terminal:
            changed = {
                field
                for field in mutable_fields
                if previous[field] != getattr(self, field)
            }
            if (
                changed == {"error_code"}
                and previous["error_code"] in _RESNAPSHOT_RECONCILING_ERROR_CODES
                and self.error_code in _RESNAPSHOT_FINAL_ERROR_CODES
            ):
                return
            if changed:
                raise ValidationError(
                    {field: "Terminal rebuild state is immutable." for field in changed}
                )
            return
        prior_signature = previous["expected_aggregate_signature"]
        if prior_signature and self.expected_aggregate_signature != prior_signature:
            raise ValidationError(
                {
                    "expected_aggregate_signature": (
                        "Expected aggregate signature is immutable once assigned."
                    )
                }
            )
        for field in (
            "completed_document_count",
            "terminal_failure_count",
            "collection_count",
            "completed_collection_count",
            "failed_collection_count",
            "enumeration_cursor",
            "expected_child_count",
            "document_publish_cursor",
        ):
            if getattr(self, field) < previous[field]:
                raise ValidationError({field: "Rebuild counters cannot decrease."})
        for field in (
            *self._ACTIVATION_AUDIT_FIELDS,
            "started_at",
            "completed_at",
        ):
            if previous[field] not in (None, "") and (
                getattr(self, field) != previous[field]
            ):
                raise ValidationError({field: "Rebuild lifecycle value is immutable."})

    def _validate_lineage_shape(self) -> None:
        predecessor = self._state.fields_cache.get("predecessor_request")
        root = self._state.fields_cache.get("lineage_root")
        if self.predecessor_request_id is None:
            if self.lineage_root_id is not None:
                raise ValidationError(
                    {"lineage_root": "Original requests cannot name a lineage root."}
                )
            return
        if self.lineage_root_id is None or self.parent_request_id is not None:
            raise ValidationError(
                {"predecessor_request": "Successor lineage shape is invalid."}
            )
        if predecessor is None:
            predecessor = (
                type(self).objects.filter(pk=self.predecessor_request_id).first()
            )
        if root is None:
            root = type(self).objects.filter(pk=self.lineage_root_id).first()
        if predecessor is None or root is None:
            raise ValidationError(
                {"predecessor_request": "Successor lineage rows must exist."}
            )
        expected_root_id = predecessor.lineage_root_id or predecessor.pk
        if (
            root.pk != expected_root_id
            or self.scope_type != predecessor.scope_type
            or self.scope_id != predecessor.scope_id
            or self.evaluation_only is not predecessor.evaluation_only
        ):
            raise ValidationError(
                {"lineage_root": "Successor must preserve its immutable scope."}
            )

    def delete(self, *args, **kwargs):
        raise ValidationError(
            {"id": "Graph rebuild request audit rows cannot be deleted."}
        )

    def _activation_audit_is_complete(self) -> bool:
        return (
            type(self.activated_artifact_pk) is int
            and self.activated_artifact_pk > 0
            and type(self.activated_run_pk) is int
            and self.activated_run_pk > 0
            and type(self.activated_build_generation) is int
            and self.activated_build_generation > 0
            and type(self.activated_build_key) is str
            and re.fullmatch(_CHECKSUM_PATTERN, self.activated_build_key) is not None
            and type(self.activated_source_hash) is str
            and re.fullmatch(_CHECKSUM_PATTERN, self.activated_source_hash) is not None
            and type(self.activated_occurrence_signature) is str
            and re.fullmatch(
                _CHECKSUM_PATTERN,
                self.activated_occurrence_signature,
            )
            is not None
        )

    def _validate_success_activation(self) -> None:
        if (
            self.status != self.Status.SUCCEEDED
            or self.scope_type == self.ScopeType.ALL
        ):
            return
        if not self._activation_audit_is_complete():
            raise ValidationError(
                {
                    "activated_occurrence_signature": (
                        "Successful scoped requests require an exact activation audit."
                    )
                }
            )
        artifact = GraphArtifact.objects.filter(pk=self.activated_artifact_pk).first()
        if artifact is None:
            previous = (
                type(self)
                .objects.filter(pk=self.pk, status=self.Status.SUCCEEDED)
                .values(*self._ACTIVATION_AUDIT_FIELDS)
                .first()
            )
            if previous is not None and all(
                previous[field] == getattr(self, field)
                for field in self._ACTIVATION_AUDIT_FIELDS
            ):
                return
            raise ValidationError(
                {"activated_artifact_pk": "Activated artifact does not exist."}
            )
        expected_scope = (
            GraphArtifact.ScopeType.DOCUMENT
            if self.scope_type == self.ScopeType.DOCUMENT
            else GraphArtifact.ScopeType.COLLECTION
        )
        if (
            artifact.rebuild_request_id != self.pk
            or artifact.scope_type != expected_scope
            or artifact.scope_id != self.scope_id
            or artifact.evaluation_only is not self.evaluation_only
        ):
            raise ValidationError(
                {"activated_artifact_pk": "Activation is outside the exact request."}
            )
        if self.scope_type == self.ScopeType.DOCUMENT:
            source_matches = len(
                self.requested_documents
            ) == 1 and artifact.source_hash == self.requested_documents[0].get(
                "source_hash"
            )
            build_kind = GraphBuildRun.BuildKind.DOCUMENT
        else:
            source_matches = (
                bool(self.expected_aggregate_signature)
                and artifact.source_hash == self.expected_aggregate_signature
            )
            build_kind = GraphBuildRun.BuildKind.COLLECTION
        if not source_matches:
            raise ValidationError(
                {"activated_artifact_pk": "Activation source differs from the request."}
            )
        runs = tuple(
            GraphBuildRun.objects.filter(
                artifact_id=artifact.pk,
                rebuild_request_id=self.pk,
                build_kind=build_kind,
            ).order_by("pk")[:2]
        )
        if len(runs) != 1:
            raise ValidationError(
                {"activated_run_pk": "Activation requires one exact terminal run."}
            )
        if runs[0].pk != self.activated_run_pk:
            raise ValidationError(
                {"activated_run_pk": "Activation run identity differs from its audit."}
            )
        from apps.knowledge_graph.services.builds import (
            _evaluation_occurrence_completed,
            _production_occurrence_completed,
        )

        completed = (
            _evaluation_occurrence_completed(
                artifact,
                runs[0],
                build_kind=build_kind,
            )
            if self.evaluation_only
            else _production_occurrence_completed(
                artifact,
                runs[0],
                build_kind=build_kind,
                allow_historical=True,
            )
        )
        if not completed:
            raise ValidationError(
                {"activated_run_pk": "Activation run is not exact and terminal."}
            )
        expected_audit = _activation_audit_values(artifact, runs[0])
        mismatched_audit = {
            field
            for field, value in expected_audit.items()
            if getattr(self, field) != value
        }
        if mismatched_audit:
            raise ValidationError(
                {
                    field: "Activation audit differs from the exact occurrence."
                    for field in sorted(mismatched_audit)
                }
            )
