"""Idempotent orchestration for document and collection graph builds.

The persistence-heavy Task 7-10 modules remain the data plane.  This module
owns only durable build identity, lifecycle, freshness fences, and scheduling.
It deliberately imports provider runtimes only through lazy stage functions.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import timedelta
from enum import StrEnum
from hashlib import sha256
from itertools import islice
from threading import Event, Thread
from time import perf_counter

import structlog
from django.core.exceptions import ValidationError
from django.db import close_old_connections, connection, transaction
from django.db.models import Count, DateTimeField, ExpressionWrapper, F, Q, Value
from django.db.models.functions import Now
from django.utils import timezone

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LEASE_DURATION = timedelta(minutes=30)
BUILD_LEASE_RETRY_SECONDS = int(_LEASE_DURATION.total_seconds()) + 30
_DOCUMENT_LOCK_NAMESPACE = 0x4B47
_REBUILD_REQUEST_LOCK_NAMESPACE = 0x4B52
_EXTRACTOR_PACKAGE_IDENTITY = "gliner2==1.3.2"
_GRAPH_TASK_PUBLISH_RETRY_POLICY = {
    "max_retries": 3,
    "interval_start": 0,
    "interval_step": 0.5,
    "interval_max": 5,
}
_ALL_REBUILD_PAGE_SIZE = 100
_MAX_REBUILD_LINEAGE_DEPTH = 32
_MAX_RESNAPSHOT_ATTEMPTS = 2
_RESNAPSHOT_PENDING_ERROR = "resnapshot_pending"
_RESNAPSHOT_CHURN_ERROR = "resnapshot_churn"
_RESNAPSHOT_RECONCILABLE_ERRORS = frozenset(
    {_RESNAPSHOT_PENDING_ERROR, _RESNAPSHOT_CHURN_ERROR}
)
_MISSING_COLLECTION_SNAPSHOT_ERROR = "rebuild collection scope no longer exists"

logger = structlog.stdlib.get_logger(__name__)


class BuildLeaseLostError(RuntimeError):
    """The caller no longer owns the durable attempt generation."""


class BuildInProgressError(RuntimeError):
    """Another live worker currently owns this exact build identity."""


class StaleBuildError(RuntimeError):
    """The immutable requested source no longer matches live source state."""


class CorruptBuildError(RuntimeError):
    """Persisted rows cannot be tied to a complete commit marker."""


class RebuildPublicationError(RuntimeError):
    """Durable request work remains resumable after broker publication failed."""

    def __init__(self, request_id: uuid.UUID, error_code: str) -> None:
        self.request_id = request_id
        self.error_code = error_code
        super().__init__(
            f"rebuild request {request_id} publication failed: {error_code}"
        )


class CommitMarkerState(StrEnum):
    """Durable stage commit state derived from both marker and persisted rows."""

    ABSENT = "absent"
    VALID = "valid"
    CORRUPT = "corrupt"


def validate_build_lease(
    run: object,
    lease_owner: str | None,
    lease_generation: int | None,
) -> None:
    """Fence every mutating stage against stale or duplicate workers."""

    if getattr(run, "orchestration_version", 0) != 1:
        return
    if type(lease_owner) is not str or not lease_owner:
        raise BuildLeaseLostError("build lease owner is required")
    if getattr(run, "lease_owner", None) != lease_owner:
        raise BuildLeaseLostError("build lease owner no longer matches")
    if type(lease_generation) is not int or isinstance(lease_generation, bool):
        raise BuildLeaseLostError("build lease generation is required")
    if getattr(run, "lease_generation", None) != lease_generation:
        raise BuildLeaseLostError("build lease generation no longer matches")
    run_id = getattr(run, "pk", None)
    if type(run_id) is not int or run_id <= 0:
        raise BuildLeaseLostError("persisted build lease row is required")
    from apps.knowledge_graph.models import GraphBuildRun

    live = GraphBuildRun.objects.filter(
        pk=run_id,
        orchestration_version=1,
        lease_owner=lease_owner,
        lease_generation=lease_generation,
        lease_expires_at__gt=Now(),
        status__in=(GraphBuildRun.Status.PENDING, GraphBuildRun.Status.RUNNING),
    ).exists()
    if not live:
        raise BuildLeaseLostError("build lease expired or no longer live")


def _lease_expiry_expression():
    return ExpressionWrapper(
        Now() + Value(_LEASE_DURATION),
        output_field=DateTimeField(),
    )


def renew_build_lease(run_id: int, lease_owner: str, lease_generation: int) -> None:
    """Renew one exact live token using the database clock."""

    from apps.knowledge_graph.models import GraphBuildRun

    if type(run_id) is not int or run_id <= 0:
        raise BuildLeaseLostError("persisted build lease row is required")
    if type(lease_owner) is not str or not lease_owner:
        raise BuildLeaseLostError("build lease owner is required")
    if type(lease_generation) is not int or isinstance(lease_generation, bool):
        raise BuildLeaseLostError("build lease generation is required")
    updated = GraphBuildRun.objects.filter(
        pk=run_id,
        orchestration_version=1,
        lease_owner=lease_owner,
        lease_generation=lease_generation,
        lease_expires_at__gt=Now(),
        status__in=(GraphBuildRun.Status.PENDING, GraphBuildRun.Status.RUNNING),
    ).update(lease_expires_at=_lease_expiry_expression())
    if updated != 1:
        raise BuildLeaseLostError("build lease expired or token was rotated")


class LeaseHeartbeat:
    """Periodically renew an exact lease while provider work is in flight."""

    def __init__(
        self,
        run_id: int,
        lease_owner: str,
        lease_generation: int,
        *,
        interval_seconds: float | None = None,
    ) -> None:
        self.run_id = run_id
        self.lease_owner = lease_owner
        self.lease_generation = lease_generation
        self.interval_seconds = (
            _LEASE_DURATION.total_seconds() / 4
            if interval_seconds is None
            else interval_seconds
        )
        if not 0 < self.interval_seconds < _LEASE_DURATION.total_seconds() / 3:
            raise ValueError("heartbeat interval must be below one-third of the lease")
        self._stop = Event()
        self._thread: Thread | None = None
        self._failure: BaseException | None = None

    def pulse(self) -> None:
        renew_build_lease(self.run_id, self.lease_owner, self.lease_generation)

    def _run(self) -> None:
        close_old_connections()
        try:
            while not self._stop.wait(self.interval_seconds):
                try:
                    self.pulse()
                except BaseException as exc:
                    self._failure = exc
                    self._stop.set()
                    return
        finally:
            close_old_connections()

    def __enter__(self):
        self.pulse()
        self._thread = Thread(
            target=self._run,
            name=f"kg-lease-{self.run_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if exc_type is None and self._failure is not None:
            raise self._failure
        return False


def _hash(value: object, label: str) -> str:
    if type(value) is not str or not _HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


class OccurrenceAction(StrEnum):
    """Bootstrap action for the newest serialized scope occurrence."""

    RETURN_ACTIVE = "return_active"
    RESUME = "resume"
    RETRY = "retry"
    CREATE = "create"


def _next_build_generation(artifacts, scope_runs=()) -> int:
    artifact_generations = tuple(
        getattr(row, "build_generation", None) for row in artifacts
    )
    run_generations = tuple(
        getattr(row, "build_generation", None) for row in scope_runs
    )
    if any(
        type(value) is not int or value < 1
        for value in (*artifact_generations, *run_generations)
    ):
        raise CorruptBuildError("build occurrence generation is invalid")
    if len(artifact_generations) != len(set(artifact_generations)):
        raise CorruptBuildError("scope owns duplicate build generations")
    if len(run_generations) != len(set(run_generations)):
        raise CorruptBuildError("scope owns duplicate run generations")
    return max((*artifact_generations, *run_generations), default=0) + 1


def _lock_latest_scope_run(build_kind: str, scope_id: object):
    """Lock the newest attached or detached audit occurrence for allocation."""

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.models.artifacts import canonical_graph_scope_id

    if build_kind not in GraphBuildRun.BuildKind.values:
        raise ValueError("build kind must be a graph scope type")
    canonical_scope = canonical_graph_scope_id(build_kind, scope_id)
    rows = tuple(
        GraphBuildRun.objects.select_for_update()
        .filter(
            build_kind=build_kind,
            scope_type=build_kind,
            scope_id=canonical_scope,
            orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        )
        .order_by("-build_generation", "-pk")[:1]
    )
    return rows


def _occurrence_action(artifacts, runs, build_key: str) -> OccurrenceAction:
    """Classify a bounded scope snapshot without collapsing A→B→A."""

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    key = _hash(build_key, "build key")
    artifact_rows = tuple(artifacts)
    run_rows = tuple(runs)
    generations = [getattr(row, "build_generation", None) for row in artifact_rows]
    if any(type(value) is not int or value < 1 for value in generations):
        raise CorruptBuildError("build occurrence generation is invalid")
    if len(generations) != len(set(generations)):
        raise CorruptBuildError("scope owns duplicate build generations")
    run_by_artifact: dict[int, object] = {}
    for run in run_rows:
        artifact_id = getattr(run, "artifact_id", None)
        if type(artifact_id) is not int or artifact_id in run_by_artifact:
            raise CorruptBuildError("artifact occurrence owns multiple build runs")
        run_by_artifact[artifact_id] = run
    exact_active = tuple(
        row
        for row in artifact_rows
        if row.build_key == key
        and row.status == GraphArtifact.Status.ACTIVE
        and getattr(row, "orchestration_version", 1) == 1
    )
    if len(exact_active) > 1:
        raise CorruptBuildError("scope owns multiple active exact-key artifacts")
    if exact_active:
        run = run_by_artifact.get(exact_active[0].pk)
        if (
            run is None
            or run.build_key != key
            or run.build_generation != exact_active[0].build_generation
            or run.stage != GraphBuildRun.Stage.ACTIVE
            or run.status != GraphBuildRun.Status.SUCCEEDED
        ):
            raise CorruptBuildError("active build occurrence is inconsistent")
        return OccurrenceAction.RETURN_ACTIVE
    if not artifact_rows:
        return OccurrenceAction.CREATE
    newest = max(artifact_rows, key=lambda row: (row.build_generation, row.pk))
    if newest.build_key != key or getattr(newest, "orchestration_version", 1) != 1:
        return OccurrenceAction.CREATE
    run = run_by_artifact.get(newest.pk)
    if (
        run is None
        or run.build_key != key
        or run.build_generation != newest.build_generation
    ):
        raise CorruptBuildError("newest build occurrence is inconsistent")
    if newest.status == GraphArtifact.Status.BUILDING and run.status in {
        GraphBuildRun.Status.PENDING,
        GraphBuildRun.Status.RUNNING,
        GraphBuildRun.Status.FAILED,
        GraphBuildRun.Status.CANCELLED,
    }:
        return OccurrenceAction.RESUME
    if newest.status in {GraphArtifact.Status.FAILED, GraphArtifact.Status.STALE}:
        return OccurrenceAction.RETRY
    return OccurrenceAction.CREATE


def _text(value: object, label: str, *, maximum: int = 512) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(character in value for character in "\x00\r\n")
    ):
        raise ValueError(f"{label} must be a safe nonempty bounded string")
    return value


@dataclass(frozen=True, slots=True)
class DocumentBuildIdentity:
    document_id: uuid.UUID
    source_hash: str
    ordered_chunk_signature: str
    extractor_package: str
    extractor_checkpoint: str
    extractor_model_revision: str
    extractor_config_checksum: str
    ontology_version: str
    ontology_checksum: str
    resolver_version: str
    resolver_checksum: str
    filter_version: str
    filter_checksum: str
    assembly_version: str
    assembly_checksum: str
    ontology_activation_signature: str = "0" * 64

    def __post_init__(self) -> None:
        if type(self.document_id) is not uuid.UUID or self.document_id.version is None:
            raise ValueError("document UUID must be an exact RFC 4122 UUID")
        for name in (
            "source_hash",
            "ordered_chunk_signature",
            "extractor_config_checksum",
            "ontology_checksum",
            "resolver_checksum",
            "filter_checksum",
            "assembly_checksum",
            "ontology_activation_signature",
        ):
            object.__setattr__(
                self,
                name,
                _hash(getattr(self, name), name.replace("_", " ")),
            )
        for name in (
            "extractor_package",
            "extractor_checkpoint",
            "extractor_model_revision",
            "ontology_version",
            "resolver_version",
            "filter_version",
            "assembly_version",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name.replace("_", " ")),
            )


@dataclass(frozen=True, slots=True)
class CollectionBuildIdentity:
    collection_id: int
    aggregate_source_signature: str
    extractor_version: str
    ontology_version: str
    ontology_checksum: str
    resolver_version: str
    resolver_checksum: str
    filter_version: str
    filter_checksum: str
    assembly_version: str
    assembly_checksum: str
    embedding_model_signature: str
    ontology_activation_signature: str = "0" * 64

    def __post_init__(self) -> None:
        if type(self.collection_id) is not int or not 0 < self.collection_id < 2**63:
            raise ValueError("collection id must be a positive database integer")
        for name in (
            "aggregate_source_signature",
            "ontology_checksum",
            "resolver_checksum",
            "filter_checksum",
            "assembly_checksum",
            "ontology_activation_signature",
        ):
            object.__setattr__(
                self,
                name,
                _hash(getattr(self, name), name.replace("_", " ")),
            )
        for name in (
            "extractor_version",
            "ontology_version",
            "resolver_version",
            "filter_version",
            "assembly_version",
            "embedding_model_signature",
        ):
            maximum = 512 if name == "embedding_model_signature" else 128
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name.replace("_", " "), maximum=maximum),
            )


def _identity_key(namespace: str, identity: object) -> str:
    payload = {
        "namespace": namespace,
        "identity": (
            asdict(identity)
            if hasattr(type(identity), "__dataclass_fields__")
            else identity
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def derive_document_build_key(identity: DocumentBuildIdentity) -> str:
    if type(identity) is not DocumentBuildIdentity:
        raise ValueError("document build identity must be exact")
    identity.__post_init__()
    return _identity_key("kg-document-build-v1", identity)


def derive_collection_build_key(identity: CollectionBuildIdentity) -> str:
    if type(identity) is not CollectionBuildIdentity:
        raise ValueError("collection build identity must be exact")
    identity.__post_init__()
    return _identity_key("kg-collection-build-v1", identity)


def ordered_chunk_signature(chunks, *, concrete_model_label: str = "") -> str:
    """Hash the exact ordered extraction inputs without retaining chunk text."""

    rows = tuple(chunks)
    payload = []
    previous_number = None
    seen_ids: set[int] = set()
    for chunk in rows:
        chunk_id = getattr(chunk, "pk", None)
        chunk_number = getattr(chunk, "chunk_number", None)
        content = getattr(chunk, "content", None)
        if (
            type(chunk_id) is not int
            or chunk_id <= 0
            or chunk_id in seen_ids
            or type(chunk_number) is not int
            or chunk_number < 0
            or (previous_number is not None and chunk_number <= previous_number)
            or type(content) is not str
        ):
            raise ValueError("ordered chunk snapshot is invalid")
        seen_ids.add(chunk_id)
        previous_number = chunk_number
        payload.append(
            {
                "chunk_id": chunk_id,
                "document_id": str(getattr(chunk, "doc_id", "")),
                "chunk_number": chunk_number,
                "start": getattr(chunk, "start_position", None),
                "end": getattr(chunk, "end_position", None),
                "modality": getattr(chunk, "modality", None),
                "content_hash": sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    return _identity_key(
        "kg-ordered-chunks-v1",
        {
            "concrete_model_label": _text(
                concrete_model_label or "unknown-document-model",
                "concrete document model",
                maximum=128,
            ),
            "chunks": tuple(payload),
        },
    )


def _extractor_version(settings: object) -> str:
    value = (
        f"{getattr(settings, 'provider', '')}:"
        f"{getattr(settings, 'model_id', '')}@"
        f"{getattr(settings, 'model_revision', '')}"
    )
    return _text(value, "extractor version", maximum=128)


@dataclass(frozen=True, slots=True)
class _DocumentContext:
    identity: DocumentBuildIdentity
    collection_id: int
    ontology: object
    settings: object


def _active_ontology(collection_id: int | None = None):
    from apps.knowledge_graph.services.ontology import (
        OntologyValidationError,
        collection_ontology,
        deployment_ontology,
    )

    try:
        return (
            deployment_ontology()
            if collection_id is None
            else collection_ontology(collection_id)
        )
    except OntologyValidationError as exc:
        raise StaleBuildError(str(exc)) from exc


def _ontology_activation_signature(ontology: object) -> str:
    """Bind build identity to an activation, including A→B→A rollbacks."""

    from apps.knowledge_graph.models import OntologyVersion

    records = tuple(
        OntologyVersion.objects.filter(
            kind=OntologyVersion.Kind.GRAPH,
            status=OntologyVersion.Status.ACTIVE,
            version=getattr(ontology, "version", None),
            checksum=getattr(ontology, "checksum", None),
        ).order_by("pk")[:2]
    )
    if len(records) != 1:
        raise StaleBuildError("graph ontology activation changed")
    record = records[0]
    return _identity_key(
        "kg-ontology-activation-v1",
        {
            "record_id": record.pk,
            "version": record.version,
            "checksum": record.checksum,
            "activated_at": (
                record.activated_at.isoformat()
                if record.activated_at is not None
                else "unrecorded"
            ),
        },
    )


def _document_context(
    document_id: object,
    expected_source_hash: object,
    *,
    for_update: bool = False,
    ontology: object | None = None,
    settings: object | None = None,
) -> _DocumentContext:
    from apps.knowledge_graph.extraction.pipeline import (
        DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS,
        DOCUMENT_EXTRACTION_V1_MAX_CHUNKS,
        DOCUMENT_EXTRACTION_V1_MAX_ENTITIES,
        DOCUMENT_EXTRACTION_V1_MAX_RELATIONS,
        _get_concrete_document,
        _ordered_chunks,
        _validate_source,
    )
    from apps.knowledge_graph.models import (
        ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM,
        ASSEMBLY_NOT_APPLICABLE_VERSION,
        graph_identity_checksum,
    )
    from apps.knowledge_graph.resolution import DOCUMENT_RESOLVER_VERSION
    from lib.knowledge_graph.config import load_extraction_settings

    if type(document_id) is not uuid.UUID:
        raise ValueError("document id must be an exact UUID")
    source_hash = _hash(expected_source_hash, "expected source hash")
    document = _get_concrete_document(document_id, for_update=for_update)
    collection_id = getattr(document, "collection_id", None)
    if type(collection_id) is not int or collection_id <= 0:
        raise StaleBuildError("document has no concrete collection membership")
    try:
        _validate_source(document, source_hash)
    except Exception as exc:
        raise StaleBuildError("document source hash changed") from exc
    chunks = _ordered_chunks(document_id, for_update=for_update)
    chunk_signature = ordered_chunk_signature(
        chunks,
        concrete_model_label=document._meta.label_lower,
    )
    ontology = _active_ontology(collection_id) if ontology is None else ontology
    settings = load_extraction_settings() if settings is None else settings
    identity = DocumentBuildIdentity(
        document_id=document_id,
        source_hash=source_hash,
        ordered_chunk_signature=chunk_signature,
        extractor_package=_EXTRACTOR_PACKAGE_IDENTITY,
        extractor_checkpoint=_text(
            getattr(settings, "model_id", None), "extractor checkpoint"
        ),
        extractor_model_revision=_text(
            getattr(settings, "model_revision", None), "extractor model revision"
        ),
        extractor_config_checksum=_identity_key(
            "kg-extractor-config-v1",
            {
                "provider": getattr(settings, "provider", None),
                "device": getattr(settings, "device", None),
                "batch_size": getattr(settings, "batch_size", None),
                "max_batch_characters": getattr(settings, "max_batch_characters", None),
                "local_files_only": getattr(settings, "local_files_only", None),
                "max_chunks": DOCUMENT_EXTRACTION_V1_MAX_CHUNKS,
                "max_characters": DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS,
                "max_entities": DOCUMENT_EXTRACTION_V1_MAX_ENTITIES,
                "max_relations": DOCUMENT_EXTRACTION_V1_MAX_RELATIONS,
            },
        ),
        ontology_version=_text(ontology.version, "ontology version", maximum=128),
        ontology_checksum=_hash(ontology.checksum, "ontology checksum"),
        resolver_version=DOCUMENT_RESOLVER_VERSION,
        resolver_checksum=graph_identity_checksum(
            "document-resolver", DOCUMENT_RESOLVER_VERSION
        ),
        filter_version="pending-v1",
        filter_checksum=graph_identity_checksum("document-filter-policy", "pending-v1"),
        assembly_version=ASSEMBLY_NOT_APPLICABLE_VERSION,
        assembly_checksum=ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM,
        ontology_activation_signature=_ontology_activation_signature(ontology),
    )
    return _DocumentContext(
        identity=identity,
        collection_id=collection_id,
        ontology=ontology,
        settings=settings,
    )


def document_graph_advisory_lock_key(document_id: uuid.UUID) -> int:
    if type(document_id) is not uuid.UUID or document_id.version is None:
        raise ValueError("document graph scope must be an exact RFC 4122 UUID")
    return (
        int.from_bytes(sha256(document_id.bytes).digest()[:4], "big", signed=False)
        & 0x7FFFFFFF
    )


def _lock_rebuild_request_creation(request_id: uuid.UUID) -> None:
    """Serialize creation/resume for one caller-supplied durable request UUID."""

    if type(request_id) is not uuid.UUID or request_id.version is None:
        raise ValueError("rebuild request id must be an exact RFC 4122 UUID")
    if connection.vendor != "postgresql":
        return
    lock_key = (
        int.from_bytes(sha256(request_id.bytes).digest()[:4], "big", signed=False)
        & 0x7FFFFFFF
    )
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_xact_lock(%s, %s)",
            [_REBUILD_REQUEST_LOCK_NAMESPACE, lock_key],
        )


def _lock_document_scope(document_id: uuid.UUID) -> None:
    if connection.vendor != "postgresql":
        return
    lock_key = document_graph_advisory_lock_key(document_id)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_xact_lock(%s, %s)",
            [_DOCUMENT_LOCK_NAMESPACE, lock_key],
        )


def lock_document_graph_advisory_scope(document_id: uuid.UUID) -> None:
    """Acquire the canonical document graph advisory transaction lock."""

    document_graph_advisory_lock_key(document_id)
    _lock_document_scope(document_id)


def _bounded_scope_artifact_ids(
    queryset,
    *,
    build_key: str,
    candidate_artifact_id: int | None = None,
) -> tuple[int, ...]:
    """Select only the current/exact/candidate occurrence ids under a scope lock."""

    from apps.knowledge_graph.models import GraphArtifact

    key = _hash(build_key, "build key")
    ids = set(
        queryset.order_by("-build_generation", "-pk").values_list("pk", flat=True)[:1]
    )
    ids.update(
        queryset.filter(status=GraphArtifact.Status.ACTIVE)
        .order_by("pk")
        .values_list("pk", flat=True)[:2]
    )
    ids.update(
        queryset.filter(
            build_key=key,
            orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        )
        .order_by("-build_generation", "-pk")
        .values_list("pk", flat=True)[:1]
    )
    ids.update(
        queryset.filter(
            build_key=key,
            status=GraphArtifact.Status.ACTIVE,
            orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        )
        .order_by("pk")
        .values_list("pk", flat=True)[:2]
    )
    if candidate_artifact_id is not None:
        candidate = (
            queryset.filter(pk=candidate_artifact_id)
            .values_list("pk", "build_generation")
            .first()
        )
        if candidate is not None:
            ids.add(candidate[0])
            ids.update(
                queryset.filter(
                    build_generation__gt=candidate[1],
                    status__in=(
                        GraphArtifact.Status.BUILDING,
                        GraphArtifact.Status.ACTIVE,
                    ),
                )
                .order_by("build_generation", "pk")
                .values_list("pk", flat=True)[:2]
            )
    return tuple(sorted(ids))


def _lock_document_build_rows(
    document_id: uuid.UUID,
    *,
    build_key: str,
    candidate_artifact_id: int | None = None,
):
    """Apply the global document lock order and return its locked rows."""

    from apps.knowledge_graph.extraction.pipeline import (
        _get_concrete_document,
        _ordered_chunks,
    )
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    _lock_document_scope(document_id)
    scope_query = GraphArtifact.objects.filter(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=str(document_id),
    )
    artifact_ids = _bounded_scope_artifact_ids(
        scope_query,
        build_key=build_key,
        candidate_artifact_id=candidate_artifact_id,
    )
    artifacts = tuple(
        GraphArtifact.objects.select_for_update()
        .filter(pk__in=artifact_ids)
        .order_by("pk")
    )
    run_ids = tuple(
        GraphBuildRun.objects.filter(
            artifact_id__in=artifact_ids,
            orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        )
        .order_by("pk")
        .values_list("pk", flat=True)[: len(artifact_ids) + 1]
    )
    runs = tuple(
        GraphBuildRun.objects.select_for_update().filter(pk__in=run_ids).order_by("pk")
    )
    document = _get_concrete_document(document_id, for_update=True)
    chunks = _ordered_chunks(document_id, for_update=True)
    return artifacts, runs, document, chunks


def _lock_terminal_document_rows(
    document_id: uuid.UUID,
    artifact_id: int,
    run_id: int,
):
    """Lock only durable orchestration rows for deletion-safe terminalization."""

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    _lock_document_scope(document_id)
    artifact = (
        GraphArtifact.objects.select_for_update()
        .filter(
            pk=artifact_id,
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id=str(document_id),
        )
        .first()
    )
    run = (
        GraphBuildRun.objects.select_for_update()
        .filter(
            pk=run_id,
            artifact_id=artifact_id,
            orchestration_version=(GraphArtifact.OrchestrationVersion.SCOPED_V1),
        )
        .first()
    )
    return artifact, run


def _lock_terminal_collection_rows(
    collection_id: int,
    artifact_id: int,
    run_id: int,
):
    """Lock a logical collection occurrence without requiring a live source row."""

    from apps.collections.models import Collection
    from apps.knowledge_graph.graph.assembly import (
        lock_collection_graph_advisory_scope,
    )
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    lock_collection_graph_advisory_scope(collection_id)
    Collection.objects.select_for_update().filter(pk=collection_id).first()
    artifact = (
        GraphArtifact.objects.select_for_update()
        .filter(
            pk=artifact_id,
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            scope_id=str(collection_id),
        )
        .first()
    )
    run = (
        GraphBuildRun.objects.select_for_update()
        .filter(
            pk=run_id,
            artifact_id=artifact_id,
            orchestration_version=(GraphArtifact.OrchestrationVersion.SCOPED_V1),
        )
        .first()
    )
    return artifact, run


def _safe_marker(value: object) -> dict[str, object]:
    return dict(value) if type(value) is dict else {}


def _commit_marker_state(
    stats: object,
    name: str,
    *,
    rows_present: bool,
    valid: bool,
) -> CommitMarkerState:
    """Classify a marker without treating falsey or empty JSON as committed."""

    marker_present = type(stats) is dict and name in stats
    if not marker_present:
        return CommitMarkerState.CORRUPT if rows_present else CommitMarkerState.ABSENT
    return CommitMarkerState.VALID if valid else CommitMarkerState.CORRUPT


def _document_extraction_commit_state(
    artifact: object,
    run: object,
) -> CommitMarkerState:
    from apps.knowledge_graph.extraction.pipeline import (
        DOCUMENT_EXTRACTION_V1_MAX_ENTITIES,
        DOCUMENT_EXTRACTION_V1_MAX_RELATIONS,
        extraction_commit_is_valid,
        extraction_evidence_fingerprint,
    )
    from apps.knowledge_graph.models import (
        EntityMention,
        GraphArtifact,
        RelationMention,
    )

    entity_query = EntityMention.objects.filter(artifact=artifact).order_by("pk")
    relation_query = RelationMention.objects.filter(artifact=artifact).order_by("pk")
    entity_count = entity_query.count()
    relation_count = relation_query.count()
    if (
        entity_count > DOCUMENT_EXTRACTION_V1_MAX_ENTITIES
        or relation_count > DOCUMENT_EXTRACTION_V1_MAX_RELATIONS
    ):
        return CommitMarkerState.CORRUPT
    evidence_fingerprint = None
    if (
        getattr(artifact, "orchestration_version", None)
        == GraphArtifact.OrchestrationVersion.SCOPED_V1
        or getattr(run, "orchestration_version", None)
        == GraphArtifact.OrchestrationVersion.SCOPED_V1
    ):
        evidence_fingerprint = extraction_evidence_fingerprint(
            entity_query,
            relation_query,
        )
    return _commit_marker_state(
        getattr(run, "stats", None),
        "extraction_commit",
        rows_present=bool(entity_count or relation_count),
        valid=extraction_commit_is_valid(
            run,
            entity_count=entity_count,
            relation_count=relation_count,
            evidence_fingerprint=evidence_fingerprint,
        ),
    )


def _document_resolution_commit_state(
    artifact: object,
    run: object,
) -> CommitMarkerState:
    from apps.knowledge_graph.models import (
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
    )
    from apps.knowledge_graph.resolution.coreference import MAX_DOCUMENT_MENTIONS
    from apps.knowledge_graph.resolution.persistence import (
        _bounded_rows,
        resolution_commit_is_valid,
        resolution_rows_fingerprint,
        source_mention_fingerprint,
    )

    stats = getattr(run, "stats", None)
    marker = stats.get("resolution_commit") if type(stats) is dict else None
    entity_query = DocumentEntity.objects.filter(artifact=artifact)
    link_query = DocumentEntityMention.objects.select_related(
        "document_entity", "mention"
    ).filter(document_entity__artifact=artifact)
    entity_count = entity_query.count()
    membership_count = link_query.count()
    mention_query = EntityMention.objects.filter(artifact=artifact).order_by("pk")
    mention_count = mention_query.count()
    rows_present = bool(entity_count or membership_count)
    if mention_count > MAX_DOCUMENT_MENTIONS:
        return CommitMarkerState.CORRUPT
    if entity_count > MAX_DOCUMENT_MENTIONS:
        return CommitMarkerState.CORRUPT
    if membership_count > MAX_DOCUMENT_MENTIONS:
        return CommitMarkerState.CORRUPT
    if type(stats) is not dict or "resolution_commit" not in stats:
        return _commit_marker_state(
            stats,
            "resolution_commit",
            rows_present=rows_present,
            valid=False,
        )
    if type(marker) is not dict:
        return CommitMarkerState.CORRUPT
    try:
        mentions = _bounded_rows(
            mention_query,
            MAX_DOCUMENT_MENTIONS,
            "document mention",
        )
        entities = _bounded_rows(
            entity_query.order_by("pk"),
            MAX_DOCUMENT_MENTIONS,
            "document resolution entity",
        )
        links = _bounded_rows(
            link_query.order_by("mention_id"),
            MAX_DOCUMENT_MENTIONS,
            "document resolution membership",
        )
        source_fingerprint = source_mention_fingerprint(mentions)
        rows_fingerprint = resolution_rows_fingerprint(entities, links)
    except (TypeError, ValueError):
        return CommitMarkerState.CORRUPT
    result_checksum = marker.get("result_checksum")
    valid_marker = resolution_commit_is_valid(
        marker,
        resolver_version=artifact.resolver_version,
        ontology_checksum=artifact.ontology_checksum,
        assembly_version=artifact.assembly_version,
        assembly_config_checksum=artifact.assembly_config_checksum,
        source_mention_count=len(mentions),
        source_mention_fingerprint=source_fingerprint,
        document_entity_count=entity_count,
        membership_count=membership_count,
        result_checksum=result_checksum,
    )
    entity_ids = {row.pk for row in entities}
    mention_ids = {row.pk for row in mentions}
    row_audits_valid = (
        len(entity_ids) == len(entities)
        and type(stats.get("resolution_rows_fingerprint")) is str
        and _HASH_PATTERN.fullmatch(stats["resolution_rows_fingerprint"]) is not None
        and stats["resolution_rows_fingerprint"] == rows_fingerprint
        and all(
            row.status == row.Status.ACTIVE
            and type(row.metadata) is dict
            and row.metadata.get("result_checksum") == result_checksum
            for row in entities
        )
        and {row.mention_id for row in links} == mention_ids
        and all(
            row.status == row.Status.ACTIVE
            and row.document_entity_id in entity_ids
            and row.mention.artifact_id == artifact.pk
            and row.resolver_version == artifact.resolver_version
            and type(row.metadata) is dict
            and row.metadata.get("result_checksum") == result_checksum
            for row in links
        )
    )
    return _commit_marker_state(
        stats,
        "resolution_commit",
        rows_present=rows_present,
        valid=bool(valid_marker and row_audits_valid),
    )


def _collection_resolution_commit_state(
    context: object,
    artifact: object,
    run: object,
    *,
    lease_owner: str,
    lease_generation: int,
) -> CommitMarkerState:
    from apps.knowledge_graph.graph.assembly import (
        CollectionGraphAssemblyError,
        CollectionGraphSourceStaleError,
        validate_collection_resolution_commit,
    )
    from apps.knowledge_graph.models import (
        CollectionEntity,
        CollectionEntityDocumentLink,
    )

    stats = getattr(run, "stats", None)
    marker_names = (
        "collection_resolution_commit",
        "filter_commit",
    )
    present = tuple(
        name for name in marker_names if type(stats) is dict and name in stats
    )
    rows_present = (
        CollectionEntity.objects.filter(artifact=artifact).exists()
        or CollectionEntityDocumentLink.objects.filter(artifact=artifact).exists()
    )
    if not present:
        return CommitMarkerState.CORRUPT if rows_present else CommitMarkerState.ABSENT
    if len(present) != 1:
        return CommitMarkerState.CORRUPT
    try:
        validate_collection_resolution_commit(
            context.identity.collection_id,
            run.pk,
            context.identity.aggregate_source_signature,
            config=context.assembly_config,
            lease_owner=lease_owner,
            lease_generation=lease_generation,
        )
    except (BuildLeaseLostError, CollectionGraphSourceStaleError):
        raise
    except CollectionGraphAssemblyError:
        return CommitMarkerState.CORRUPT
    return CommitMarkerState.VALID


def _collection_assembly_commit_state(
    context: object,
    artifact: object,
    run: object,
    *,
    lease_owner: str,
    lease_generation: int,
) -> CommitMarkerState:
    from apps.knowledge_graph.graph.assembly import (
        CollectionGraphAssemblyError,
        CollectionGraphSourceStaleError,
        validate_collection_graph_artifact,
    )
    from apps.knowledge_graph.models import (
        CollectionRelation,
        CollectionRelationEvidence,
    )

    stats = getattr(run, "stats", None)
    rows_present = (
        CollectionRelation.objects.filter(artifact=artifact).exists()
        or CollectionRelationEvidence.objects.filter(artifact=artifact).exists()
    )
    if type(stats) is not dict or "collection_assembly_commit" not in stats:
        return _commit_marker_state(
            stats,
            "collection_assembly_commit",
            rows_present=rows_present,
            valid=False,
        )
    try:
        validate_collection_graph_artifact(
            context.identity.collection_id,
            run.pk,
            context.identity.aggregate_source_signature,
            ontology=context.ontology,
            config=context.assembly_config,
            lease_owner=lease_owner,
            lease_generation=lease_generation,
        )
    except (BuildLeaseLostError, CollectionGraphSourceStaleError):
        raise
    except CollectionGraphAssemblyError:
        return CommitMarkerState.CORRUPT
    return CommitMarkerState.VALID


def _attempt_history(run: object) -> list[dict[str, object]]:
    metadata = _safe_marker(getattr(run, "metadata", None))
    raw = metadata.get("attempt_history", [])
    history = list(raw[-31:]) if type(raw) is list else []
    history.append(
        {
            "attempt": int(getattr(run, "attempt", 1)),
            "stage": str(getattr(run, "stage", "")),
            "status": str(getattr(run, "status", "")),
            "error_code": str(getattr(run, "error_code", ""))[:128],
        }
    )
    return history[-32:]


def _claim_locked_run(run: object, owner: str) -> tuple[str, int]:
    from apps.knowledge_graph.models import GraphBuildRun

    if type(owner) is not str or not owner:
        raise ValueError("lease owner must be a nonempty string")
    updated = (
        GraphBuildRun.objects.filter(
            pk=run.pk,
            orchestration_version=1,
            status__in=(GraphBuildRun.Status.PENDING, GraphBuildRun.Status.RUNNING),
        )
        .filter(Q(lease_owner="") | Q(lease_expires_at__lte=Now()))
        .update(
            lease_owner=owner,
            lease_generation=F("lease_generation") + 1,
            lease_expires_at=_lease_expiry_expression(),
        )
    )
    if updated != 1:
        raise BuildInProgressError("exact graph build already has a live lease")
    run.refresh_from_db(fields=["lease_owner", "lease_generation", "lease_expires_at"])
    return owner, run.lease_generation


def _run_has_live_lease(run: object) -> bool:
    from apps.knowledge_graph.models import GraphBuildRun

    return GraphBuildRun.objects.filter(
        pk=run.pk,
        orchestration_version=1,
        lease_owner__gt="",
        lease_expires_at__gt=Now(),
        status__in=(GraphBuildRun.Status.PENDING, GraphBuildRun.Status.RUNNING),
    ).exists()


def _restart_locked_run(run: object) -> None:
    from apps.knowledge_graph.models import GraphBuildRun

    if getattr(run, "orchestration_version", None) != 1:
        raise CorruptBuildError("only a typed scoped run can be restarted")
    try:
        validate_orchestration_stage(run.build_kind, run.stage, run.status)
    except ValidationError as exc:
        raise CorruptBuildError("build restart state is invalid") from exc
    if run.status not in {
        GraphBuildRun.Status.FAILED,
        GraphBuildRun.Status.CANCELLED,
        GraphBuildRun.Status.PENDING,
        GraphBuildRun.Status.RUNNING,
    }:
        raise CorruptBuildError("completed build cannot be restarted")
    metadata = _safe_marker(run.metadata)
    metadata["attempt_history"] = _attempt_history(run)
    GraphBuildRun.objects.filter(pk=run.pk).update(
        attempt=run.attempt + 1,
        stage=GraphBuildRun.Stage.QUEUED,
        status=GraphBuildRun.Status.PENDING,
        error_code="",
        error_message="",
        error_metadata={},
        metadata=metadata,
        started_at=None,
        finished_at=None,
        lease_owner="",
        lease_expires_at=None,
    )
    run.refresh_from_db()


def _validate_retryable_run(run: object) -> None:
    if getattr(run, "error_code", "") == "corrupt_build_state":
        raise CorruptBuildError(
            "corrupt build occurrence is permanently failed and cannot be retried"
        )


def _transition_run(
    run_id: int,
    target: str,
    *,
    lease_owner: str,
    lease_generation: int,
    marker: dict[str, object] | None = None,
) -> object:
    from apps.knowledge_graph.models import GraphBuildRun

    started = perf_counter()
    with transaction.atomic():
        run = GraphBuildRun.objects.select_for_update().get(pk=run_id)
        validate_build_lease(run, lease_owner, lease_generation)
        validate_stage_transition(run.build_kind, run.stage, target)
        status = {
            GraphBuildRun.Stage.QUEUED: GraphBuildRun.Status.PENDING,
            GraphBuildRun.Stage.EXTRACTING: GraphBuildRun.Status.RUNNING,
            GraphBuildRun.Stage.SNAPSHOTTING: GraphBuildRun.Status.RUNNING,
            GraphBuildRun.Stage.RESOLVING: GraphBuildRun.Status.RUNNING,
            GraphBuildRun.Stage.ASSEMBLING: GraphBuildRun.Status.RUNNING,
            GraphBuildRun.Stage.VALIDATING: GraphBuildRun.Status.RUNNING,
            GraphBuildRun.Stage.ACTIVE: GraphBuildRun.Status.SUCCEEDED,
            GraphBuildRun.Stage.FAILED: GraphBuildRun.Status.FAILED,
            GraphBuildRun.Stage.SUPERSEDED: GraphBuildRun.Status.CANCELLED,
            GraphBuildRun.Stage.STALE: GraphBuildRun.Status.CANCELLED,
        }[target]
        now = timezone.now()
        run.stage = target
        run.status = status
        if run.started_at is None and status == GraphBuildRun.Status.RUNNING:
            run.started_at = now
        terminal = status in {
            GraphBuildRun.Status.SUCCEEDED,
            GraphBuildRun.Status.FAILED,
            GraphBuildRun.Status.CANCELLED,
        }
        if terminal:
            run.finished_at = now
            run.lease_owner = ""
            run.lease_expires_at = None
        stage_marker = _safe_marker(run.stage_marker)
        sequence = stage_marker.get("stage_sequence", [])
        sequence = list(sequence[-31:]) if type(sequence) is list else []
        sequence.append(target)
        stage_marker["stage_sequence"] = sequence[-32:]
        stage_marker["last_stage"] = target
        if marker:
            stage_marker.update(marker)
        run.stage_marker = stage_marker
        run.save(
            update_fields=[
                "stage",
                "status",
                "started_at",
                "finished_at",
                "lease_owner",
                "lease_expires_at",
                "stage_marker",
            ]
        )
        if not terminal:
            renew_build_lease(run.pk, lease_owner, lease_generation)
            run.refresh_from_db(fields=["lease_expires_at"])
    logger.info(
        "obs.kg.build_stage",
        build_kind=run.build_kind,
        scope_id=run.scope_id,
        build_key=run.build_key,
        build_run_id=run.pk,
        artifact_id=run.artifact_id,
        attempt=run.attempt,
        stage=target,
        stage_seconds=perf_counter() - started,
    )
    return run


def _transition_table(build_kind: str) -> dict[str, frozenset[str]]:
    from apps.knowledge_graph.models import GraphBuildRun

    common_failures = frozenset({GraphBuildRun.Stage.FAILED, GraphBuildRun.Stage.STALE})
    if build_kind == GraphBuildRun.BuildKind.DOCUMENT:
        return {
            GraphBuildRun.Stage.QUEUED: frozenset({GraphBuildRun.Stage.EXTRACTING})
            | common_failures,
            GraphBuildRun.Stage.EXTRACTING: frozenset({GraphBuildRun.Stage.RESOLVING})
            | common_failures,
            GraphBuildRun.Stage.RESOLVING: frozenset({GraphBuildRun.Stage.VALIDATING})
            | common_failures,
            GraphBuildRun.Stage.VALIDATING: frozenset(
                {GraphBuildRun.Stage.ACTIVE, GraphBuildRun.Stage.SUPERSEDED}
            )
            | common_failures,
            GraphBuildRun.Stage.ACTIVE: frozenset(
                {GraphBuildRun.Stage.SUPERSEDED, GraphBuildRun.Stage.STALE}
            ),
            GraphBuildRun.Stage.FAILED: frozenset(),
            GraphBuildRun.Stage.SUPERSEDED: frozenset(),
            GraphBuildRun.Stage.STALE: frozenset(),
        }
    if build_kind == GraphBuildRun.BuildKind.COLLECTION:
        return {
            GraphBuildRun.Stage.QUEUED: frozenset({GraphBuildRun.Stage.SNAPSHOTTING})
            | common_failures,
            GraphBuildRun.Stage.SNAPSHOTTING: frozenset({GraphBuildRun.Stage.RESOLVING})
            | common_failures,
            GraphBuildRun.Stage.RESOLVING: frozenset({GraphBuildRun.Stage.ASSEMBLING})
            | common_failures,
            GraphBuildRun.Stage.ASSEMBLING: frozenset({GraphBuildRun.Stage.VALIDATING})
            | common_failures,
            GraphBuildRun.Stage.VALIDATING: frozenset(
                {GraphBuildRun.Stage.ACTIVE, GraphBuildRun.Stage.SUPERSEDED}
            )
            | common_failures,
            GraphBuildRun.Stage.ACTIVE: frozenset(
                {GraphBuildRun.Stage.SUPERSEDED, GraphBuildRun.Stage.STALE}
            ),
            GraphBuildRun.Stage.FAILED: frozenset(),
            GraphBuildRun.Stage.SUPERSEDED: frozenset(),
            GraphBuildRun.Stage.STALE: frozenset(),
        }
    raise ValidationError({"build_kind": "Unknown graph build kind."})


def validate_stage_transition(build_kind: str, current: str, target: str) -> None:
    table = _transition_table(build_kind)
    if current == target:
        if current not in table:
            raise ValidationError({"stage": "Unknown orchestration stage."})
        return
    if target not in table.get(current, frozenset()):
        raise ValidationError(
            {"stage": f"Invalid {build_kind} graph stage transition."}
        )


def validate_orchestration_stage(build_kind: str, stage: str, status: str) -> None:
    from apps.knowledge_graph.models import GraphBuildRun

    table = _transition_table(build_kind)
    if stage not in table:
        raise ValidationError({"stage": "Stage is not valid for this build kind."})
    expected_status = {
        GraphBuildRun.Stage.QUEUED: GraphBuildRun.Status.PENDING,
        GraphBuildRun.Stage.EXTRACTING: GraphBuildRun.Status.RUNNING,
        GraphBuildRun.Stage.SNAPSHOTTING: GraphBuildRun.Status.RUNNING,
        GraphBuildRun.Stage.RESOLVING: GraphBuildRun.Status.RUNNING,
        GraphBuildRun.Stage.ASSEMBLING: GraphBuildRun.Status.RUNNING,
        GraphBuildRun.Stage.VALIDATING: GraphBuildRun.Status.RUNNING,
        GraphBuildRun.Stage.ACTIVE: GraphBuildRun.Status.SUCCEEDED,
        GraphBuildRun.Stage.FAILED: GraphBuildRun.Status.FAILED,
        GraphBuildRun.Stage.SUPERSEDED: GraphBuildRun.Status.CANCELLED,
        GraphBuildRun.Stage.STALE: GraphBuildRun.Status.CANCELLED,
    }[stage]
    if status != expected_status:
        raise ValidationError(
            {"status": "Build status does not match its orchestration stage."}
        )


def _publish_graph_task(
    task,
    *,
    kwargs: dict[str, object],
    build_kind: str,
    scope_id: str,
) -> None:
    try:
        task.apply_async(
            kwargs=kwargs,
            retry=True,
            retry_policy=dict(_GRAPH_TASK_PUBLISH_RETRY_POLICY),
        )
    except Exception as exc:
        logger.error(
            "obs.kg.task_publish_failed",
            task_name=task.name,
            build_kind=build_kind,
            scope_id=scope_id,
            error_type=type(exc).__name__,
            publish_retry_exhausted=True,
            durable_outbox=False,
        )
        raise


def derive_current_document_build_key(
    document_id: uuid.UUID,
    expected_source_hash: str,
) -> str:
    """Derive the exact immutable Task 11 key for a current document snapshot."""

    context = _document_context(document_id, expected_source_hash)
    return derive_document_build_key(context.identity)


_MAX_REBUILD_DOCUMENTS = 10_000


def assert_evaluation_bypass(eval_only: bool) -> bool:
    """Independently authorize the deliberately narrow build-disabled bypass."""

    if type(eval_only) is not bool:
        raise ValueError("evaluation marker must be an exact boolean")
    if not eval_only:
        return False
    from django.conf import settings

    explicit = getattr(settings, "KG_EVAL_BYPASS_ALLOWED", False) is True
    nonproduction = (
        getattr(settings, "DEBUG", False) is True
        or getattr(settings, "TESTING", False) is True
    )
    if not explicit or not nonproduction:
        raise PermissionError(
            "evaluation-only graph bypass is not authorized in this environment"
        )
    return True


def _canonical_request_uuid(
    value: object, *, required: bool = False
) -> uuid.UUID | None:
    if value is None:
        if required:
            raise ValueError("rebuild request id is required")
        return None
    if type(value) is uuid.UUID:
        if value.version is None:
            raise ValueError("rebuild request id must be an RFC 4122 UUID")
        return value
    if type(value) is not str:
        raise ValueError("rebuild request id must be a canonical UUID string")
    try:
        resolved = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("rebuild request id must be a canonical UUID string") from exc
    if str(resolved) != value:
        raise ValueError("rebuild request id must be a canonical UUID string")
    return resolved


def _lock_rebuild_request_prefix(request_id: uuid.UUID):
    """Lock hierarchy parent, lineage root, then successors in one order."""

    from apps.knowledge_graph.models import GraphRebuildRequest

    metadata = (
        GraphRebuildRequest.objects.filter(pk=request_id)
        .values("pk", "lineage_root_id", "parent_request_id")
        .first()
    )
    if metadata is None:
        raise LookupError("rebuild request does not exist")
    root_id = metadata["lineage_root_id"] or metadata["pk"]
    root_metadata = (
        GraphRebuildRequest.objects.filter(pk=root_id)
        .values("pk", "parent_request_id")
        .first()
    )
    if root_metadata is None:
        raise CorruptBuildError("rebuild lineage root is missing")
    hierarchy_parent = None
    parent_id = root_metadata["parent_request_id"]
    if parent_id is not None:
        hierarchy_parent = GraphRebuildRequest.objects.select_for_update(
            no_key=True
        ).get(pk=parent_id)
    current = GraphRebuildRequest.objects.select_for_update(no_key=True).get(pk=root_id)
    for _ in range(_MAX_REBUILD_LINEAGE_DEPTH):
        if current.pk == request_id:
            return hierarchy_parent, current
        successor = (
            GraphRebuildRequest.objects.select_for_update(no_key=True)
            .filter(predecessor_request_id=current.pk)
            .first()
        )
        if successor is None:
            raise CorruptBuildError("rebuild successor lineage is incomplete")
        current = successor
    raise CorruptBuildError("rebuild successor lineage exceeds its cap")


def _effective_rebuild_request(request: object, *, for_update: bool = False):
    """Follow one immutable successor chain to its current effective request."""

    from apps.knowledge_graph.models import GraphRebuildRequest

    current = request
    for _ in range(_MAX_REBUILD_LINEAGE_DEPTH):
        query = GraphRebuildRequest.objects
        if for_update:
            query = query.select_for_update()
        successor = query.filter(predecessor_request_id=current.pk).first()
        if successor is None:
            return current
        current = successor
    raise CorruptBuildError("rebuild successor lineage exceeds its cap")


def _validated_rebuild_request_before_content(
    request_id: object,
    eval_only: bool,
    *,
    build_kind: str,
    scope_id: object,
    source_hash: object,
):
    """Resolve request metadata and authorize its exact marker before content reads."""

    assert_evaluation_bypass(eval_only)
    return validate_rebuild_task_request_metadata(
        request_id,
        eval_only,
        build_kind=build_kind,
        scope_id=scope_id,
        source_hash=source_hash,
        _authorize_stored_evaluation=True,
    )


def validate_rebuild_task_request_metadata(
    request_id: object,
    eval_only: bool,
    *,
    build_kind: str,
    scope_id: object,
    source_hash: object,
    _authorize_stored_evaluation: bool = False,
):
    """Validate exact request correlation using metadata only.

    This deliberately does not authorize the evaluation bypass.  Task boundaries
    use it after validating their own marker so malformed or disabled deliveries
    can only terminalize the exact request they belong to.  No document or
    collection content is read here.
    """

    if type(eval_only) is not bool:
        raise ValueError("evaluation marker must be an exact boolean")
    if build_kind == "document":
        if type(scope_id) is not uuid.UUID:
            raise ValueError("document id must be an exact UUID")
    elif build_kind == "collection":
        if type(scope_id) is not int or not 0 < scope_id < 2**63:
            raise ValueError("collection id must be a positive database integer")
    else:
        raise ValueError("build kind must be document or collection")
    source_hash = _hash(source_hash, f"{build_kind} source hash")
    resolved_id = _canonical_request_uuid(request_id, required=eval_only)
    if resolved_id is None:
        return None
    from apps.knowledge_graph.models import GraphRebuildRequest

    request = GraphRebuildRequest.objects.filter(pk=resolved_id).first()
    if request is None:
        raise StaleBuildError("rebuild request no longer exists")
    if _authorize_stored_evaluation and request.evaluation_only:
        assert_evaluation_bypass(True)
    if request.evaluation_only is not eval_only:
        raise CorruptBuildError("rebuild evaluation marker changed")
    if request.status not in {
        GraphRebuildRequest.Status.RUNNING,
        GraphRebuildRequest.Status.SUCCEEDED,
    }:
        raise StaleBuildError("rebuild request is already terminal")
    if build_kind == "collection":
        if (
            request.scope_type != GraphRebuildRequest.ScopeType.COLLECTION
            or request.scope_id != str(scope_id)
            or request.expected_aggregate_signature != source_hash
        ):
            raise StaleBuildError("collection is outside the rebuild request scope")
        return request
    matching = tuple(
        row
        for row in request.requested_documents
        if row.get("document_id") == str(scope_id)
        and row.get("source_hash") == source_hash
    )
    if len(matching) != 1:
        raise StaleBuildError("document is outside the rebuild request snapshot")
    if request.scope_type == GraphRebuildRequest.ScopeType.DOCUMENT:
        valid_scope = request.scope_id == str(scope_id)
    elif request.scope_type == GraphRebuildRequest.ScopeType.COLLECTION:
        valid_scope = matching[0].get("collection_id") == int(request.scope_id)
    else:
        valid_scope = False
    if not valid_scope:
        raise StaleBuildError("document is outside the rebuild request scope")
    return request


def _snapshot_rebuild_documents(
    *,
    scope_type: str,
    scope_id: object,
    for_update: bool,
) -> tuple[dict[str, object], ...]:
    """Capture only immutable scalar document identity, never document content."""

    from apps.collections.models import Collection
    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    if scope_type == "document":
        if type(scope_id) is not uuid.UUID:
            raise ValueError("document rebuild scope must be an exact UUID")
    elif scope_type == "collection":
        if type(scope_id) is not int or not 0 < scope_id < 2**63:
            raise ValueError("collection rebuild scope must be a positive integer")
        collection_query = Collection.objects
        if for_update:
            collection_query = collection_query.select_for_update()
        if not collection_query.filter(pk=scope_id).exists():
            raise LookupError("collection does not exist")
    else:
        raise ValueError("rebuild scope must be document or collection")

    snapshots: list[dict[str, object]] = []
    for model in sorted(
        DESCENDED_FROM_DOCUMENT, key=lambda item: item._meta.label_lower
    ):
        query = model.objects.filter(ingestion_complete=True)
        if scope_type == "document":
            query = query.filter(id=scope_id)
        else:
            query = query.filter(collection_id=scope_id)
        if for_update:
            query = query.select_for_update()
        lock_order = ("pkid", "id") if for_update else ("id", "pkid")
        rows = query.order_by(*lock_order).values(
            "id", "pkid", "collection_id", "full_text_hash"
        )
        for row in islice(
            rows.iterator(chunk_size=1_000),
            _MAX_REBUILD_DOCUMENTS - len(snapshots) + 1,
        ):
            if len(snapshots) >= _MAX_REBUILD_DOCUMENTS:
                raise RuntimeError("rebuild document snapshot exceeds its cap")
            snapshots.append(
                {
                    "document_id": str(row["id"]),
                    "document_pkid": int(row["pkid"]),
                    "model_label": model._meta.label_lower,
                    "collection_id": int(row["collection_id"]),
                    "source_hash": _hash(row["full_text_hash"], "document source hash"),
                }
            )
    snapshots.sort(
        key=lambda row: (
            row["model_label"],
            row["document_id"],
            row["document_pkid"],
        )
    )
    ids = tuple(row["document_id"] for row in snapshots)
    if len(ids) != len(set(ids)):
        raise CorruptBuildError("rebuild scope contains duplicate document UUIDs")
    if scope_type == "document" and len(snapshots) != 1:
        raise LookupError("document rebuild scope is missing or ambiguous")
    return tuple(snapshots)


def _lock_snapshot_concrete_rows(
    snapshots: tuple[dict[str, object], ...],
) -> None:
    from collections import defaultdict

    from django.apps import apps as django_apps

    refs_by_model: dict[str, dict[tuple[int, uuid.UUID], None]] = defaultdict(dict)
    for row in snapshots:
        model_label = row.get("model_label")
        document_pkid = row.get("document_pkid")
        document_id = row.get("document_id")
        if (
            type(model_label) is not str
            or type(document_pkid) is not int
            or type(document_id) is not str
        ):
            raise CorruptBuildError("rebuild document snapshot identity is invalid")
        refs_by_model[model_label][(document_pkid, uuid.UUID(document_id))] = None
    for model_label in sorted(refs_by_model):
        model = django_apps.get_model(model_label)
        refs = tuple(
            sorted(
                refs_by_model[model_label],
                key=lambda row: (row[0], row[1].int),
            )
        )
        for offset in range(0, len(refs), 500):
            batch = refs[offset : offset + 500]
            tuple(
                model._base_manager.select_for_update()
                .filter(pkid__in=tuple(row[0] for row in batch))
                .order_by("pkid", "id")
            )


def _lock_scoped_rebuild_snapshot(
    *, scope_type: str, scope_id: object
) -> tuple[dict[str, object], ...]:
    """Capture a new request under the canonical C→D→document lock prefix."""

    from apps.collections.models import Collection
    from apps.knowledge_graph.graph.assembly import (
        lock_collection_graph_advisory_scope,
    )

    prelock = _snapshot_rebuild_documents(
        scope_type=scope_type,
        scope_id=scope_id,
        for_update=False,
    )
    collection_ids = {
        int(row["collection_id"])
        for row in prelock
        if type(row.get("collection_id")) is int
    }
    if scope_type == "collection":
        collection_ids.add(int(scope_id))
    for collection_id in sorted(collection_ids):
        lock_collection_graph_advisory_scope(collection_id)
    locked_collection_ids = tuple(
        Collection.objects.select_for_update()
        .filter(pk__in=tuple(sorted(collection_ids)))
        .order_by("pk")
        .values_list("pk", flat=True)
    )
    if locked_collection_ids != tuple(sorted(collection_ids)):
        raise StaleBuildError("rebuild collection scope no longer exists")
    fenced = _snapshot_rebuild_documents(
        scope_type=scope_type,
        scope_id=scope_id,
        for_update=False,
    )
    if any(int(row["collection_id"]) not in collection_ids for row in fenced):
        raise StaleBuildError("rebuild collection lock set changed")
    document_ids = {uuid.UUID(str(row["document_id"])) for row in (*prelock, *fenced)}
    for document_id in sorted(
        document_ids,
        key=lambda value: (document_graph_advisory_lock_key(value), value.int),
    ):
        _lock_document_scope(document_id)
    _lock_snapshot_concrete_rows((*prelock, *fenced))
    current = _snapshot_rebuild_documents(
        scope_type=scope_type,
        scope_id=scope_id,
        for_update=False,
    )
    if current != fenced:
        raise StaleBuildError("rebuild snapshot changed under its canonical locks")
    return current


def _lock_request_completion_snapshot(
    request: object,
    expected: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    """Lock one request's C→D→artifact/run→document completion prefix."""

    from apps.collections.models import Collection
    from apps.knowledge_graph.graph.assembly import (
        lock_collection_graph_advisory_scope,
    )
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    scope_type = request.scope_type
    scope_id: object = (
        uuid.UUID(request.scope_id)
        if scope_type == "document"
        else int(request.scope_id)
    )
    prelock = _snapshot_rebuild_documents(
        scope_type=scope_type,
        scope_id=scope_id,
        for_update=False,
    )
    collection_ids = {
        int(row["collection_id"])
        for row in (*expected, *prelock)
        if type(row.get("collection_id")) is int
    }
    if scope_type == "collection":
        collection_ids.add(int(request.scope_id))
    for collection_id in sorted(collection_ids):
        lock_collection_graph_advisory_scope(collection_id)
    locked_collection_ids = tuple(
        Collection.objects.select_for_update()
        .filter(pk__in=tuple(sorted(collection_ids)))
        .order_by("pk")
        .values_list("pk", flat=True)
    )
    if locked_collection_ids != tuple(sorted(collection_ids)):
        raise StaleBuildError("rebuild collection scope no longer exists")

    fenced = _snapshot_rebuild_documents(
        scope_type=scope_type,
        scope_id=scope_id,
        for_update=False,
    )
    document_ids = {uuid.UUID(str(row["document_id"])) for row in (*expected, *fenced)}
    for document_id in sorted(
        document_ids,
        key=lambda value: (document_graph_advisory_lock_key(value), value.int),
    ):
        _lock_document_scope(document_id)

    artifact_rows = tuple(
        GraphArtifact.objects.select_for_update(no_key=True)
        .filter(
            rebuild_request_id=request.pk,
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
        )
        .order_by("pk")[: len(expected) + 1]
    )
    if len(artifact_rows) > len(expected):
        raise CorruptBuildError("request owns surplus document artifacts")
    run_rows = tuple(
        GraphBuildRun.objects.select_for_update()
        .filter(
            rebuild_request_id=request.pk,
            build_kind=GraphBuildRun.BuildKind.DOCUMENT,
        )
        .order_by("pk")[: len(expected) + 1]
    )
    if len(run_rows) > len(expected):
        raise CorruptBuildError("request owns surplus document build runs")

    _lock_snapshot_concrete_rows((*expected, *fenced))

    current = _snapshot_rebuild_documents(
        scope_type=scope_type,
        scope_id=scope_id,
        for_update=False,
    )
    if current != fenced:
        raise CorruptBuildError("rebuild snapshot changed under its canonical locks")
    return current


def preview_rebuild(*, scope_type: str, scope_id: object) -> dict[str, int]:
    """Return bounded counts without creating requests or publishing tasks."""

    from apps.collections.models import Collection
    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    if scope_type in {"document", "collection"}:
        snapshots = _snapshot_rebuild_documents(
            scope_type=scope_type,
            scope_id=scope_id,
            for_update=False,
        )
        return {
            "document_count": len(snapshots),
            "collection_count": 1 if scope_type == "collection" else 0,
        }
    if scope_type != "all" or scope_id is not None:
        raise ValueError("operator-wide rebuild requires an empty scope id")
    collection_count = Collection.objects.count()
    document_count = 0
    for model in DESCENDED_FROM_DOCUMENT:
        document_count += model.objects.filter(ingestion_complete=True).count()
    return {
        "document_count": document_count,
        "collection_count": collection_count,
    }


def _create_scoped_rebuild_request(
    *,
    scope_type: str,
    scope_id: object,
    request_id: uuid.UUID,
    evaluation_only: bool,
    parent_request: object | None,
    predecessor_request: object | None = None,
    lineage_root: object | None = None,
):
    from apps.knowledge_graph.models import GraphRebuildRequest

    snapshots = _lock_scoped_rebuild_snapshot(
        scope_type=scope_type,
        scope_id=scope_id,
    )
    request = GraphRebuildRequest.objects.create(
        id=request_id,
        parent_request=parent_request,
        predecessor_request=predecessor_request,
        lineage_root=lineage_root,
        scope_type=scope_type,
        scope_id=str(scope_id),
        requested_documents=list(snapshots),
        document_count=len(snapshots),
        collection_count=(
            1 if scope_type == GraphRebuildRequest.ScopeType.COLLECTION else 0
        ),
        document_publication_state=(GraphRebuildRequest.PublicationState.PENDING),
        collection_publication_state=(
            GraphRebuildRequest.PublicationState.NOT_APPLICABLE
        ),
        evaluation_only=evaluation_only,
        status=GraphRebuildRequest.Status.RUNNING,
        started_at=timezone.now(),
    )
    return request


def _publish_rebuild_document_tasks(
    request_id: uuid.UUID, *, raise_on_failure: bool = True
) -> bool:
    """At-least-once publish one immutable snapshot from its durable cursor."""

    from apps.knowledge_graph.models import GraphRebuildRequest

    request = GraphRebuildRequest.objects.filter(pk=request_id).first()
    if request is None:
        return True
    request = _effective_rebuild_request(request)
    while request.status == GraphRebuildRequest.Status.RUNNING:
        cursor = request.document_publish_cursor
        if cursor >= request.document_count:
            with transaction.atomic():
                _parent, locked = _lock_rebuild_request_prefix(request.pk)
                if locked.status == GraphRebuildRequest.Status.RUNNING:
                    locked.document_publication_state = (
                        GraphRebuildRequest.PublicationState.PUBLISHED
                    )
                    if locked.error_code == "document_task_publish_failed":
                        locked.error_code = ""
                    locked.save(
                        update_fields=[
                            "document_publication_state",
                            "error_code",
                            "updated_at",
                        ]
                    )
            if request.document_count == 0:
                advance_rebuild_request(request.pk)
            return True
        snapshot = request.requested_documents[cursor]
        try:
            enqueue_document_build(
                uuid.UUID(snapshot["document_id"]),
                snapshot["source_hash"],
                request_id=request.pk,
                eval_only=request.evaluation_only,
            )
        except Exception:
            with transaction.atomic():
                _parent, locked = _lock_rebuild_request_prefix(request.pk)
                if locked.status == GraphRebuildRequest.Status.RUNNING:
                    locked.document_publication_state = (
                        GraphRebuildRequest.PublicationState.FAILED
                    )
                    locked.error_code = "document_task_publish_failed"
                    locked.save(
                        update_fields=[
                            "document_publication_state",
                            "error_code",
                            "updated_at",
                        ]
                    )
            if raise_on_failure:
                raise RebuildPublicationError(
                    request.pk,
                    "document_task_publish_failed",
                ) from None
            return False
        with transaction.atomic():
            _parent, locked = _lock_rebuild_request_prefix(request.pk)
            publication_finished = False
            if (
                locked.status == GraphRebuildRequest.Status.RUNNING
                and locked.document_publish_cursor == cursor
            ):
                locked.document_publish_cursor = cursor + 1
                locked.document_publication_state = (
                    GraphRebuildRequest.PublicationState.PUBLISHED
                    if cursor + 1 == locked.document_count
                    else GraphRebuildRequest.PublicationState.PENDING
                )
                publication_finished = cursor + 1 == locked.document_count
                if locked.error_code == "document_task_publish_failed":
                    locked.error_code = ""
                locked.save(
                    update_fields=[
                        "document_publish_cursor",
                        "document_publication_state",
                        "error_code",
                        "updated_at",
                    ]
                )
            request = locked
        if publication_finished:
            advance_rebuild_request(request.pk)
            return True
    return True


def _publish_operator_rebuild_children(parent_request_id: uuid.UUID) -> None:
    """Resume every committed child without retaining an unbounded callback."""

    from django.db.models import BigIntegerField
    from django.db.models.functions import Cast

    from apps.knowledge_graph.models import GraphRebuildRequest

    child_ids = (
        GraphRebuildRequest.objects.filter(parent_request_id=parent_request_id)
        .annotate(numeric_scope_id=Cast("scope_id", BigIntegerField()))
        .order_by("numeric_scope_id", "pk")
        .values_list("pk", flat=True)
        .iterator(chunk_size=500)
    )
    for child_id in child_ids:
        resume_rebuild_request(child_id)


def _enumerate_operator_rebuild_page(
    parent_request_id: uuid.UUID,
) -> tuple[uuid.UUID, ...]:
    """Commit one parent→Collection→Document page and its deterministic children."""

    from apps.collections.models import Collection
    from apps.knowledge_graph.graph.assembly import (
        lock_collection_graph_advisory_scope,
    )
    from apps.knowledge_graph.models import GraphRebuildRequest

    with transaction.atomic():
        parent = GraphRebuildRequest.objects.select_for_update().get(
            pk=parent_request_id
        )
        if parent.scope_type != GraphRebuildRequest.ScopeType.ALL:
            raise CorruptBuildError("operator rebuild parent scope changed")
        if parent.enumeration_complete:
            return ()
        high_water = parent.enumeration_high_water
        if type(high_water) is not int or high_water < 0:
            raise CorruptBuildError("operator rebuild high-water is invalid")
        collection_ids = tuple(
            Collection.objects.filter(
                pk__gt=parent.enumeration_cursor,
                pk__lte=high_water,
            )
            .order_by("pk")
            .values_list("pk", flat=True)[:_ALL_REBUILD_PAGE_SIZE]
        )
        if not collection_ids:
            parent.enumeration_cursor = high_water
            parent.enumeration_complete = True
            parent.save(
                update_fields=[
                    "enumeration_cursor",
                    "enumeration_complete",
                    "updated_at",
                ]
            )
            return ()
        for collection_id in collection_ids:
            lock_collection_graph_advisory_scope(collection_id)
        locked_collection_ids = tuple(
            Collection.objects.select_for_update()
            .filter(pk__in=collection_ids)
            .order_by("pk")
            .values_list("pk", flat=True)
        )
        if locked_collection_ids != collection_ids:
            raise StaleBuildError(
                "operator rebuild collection page changed during locking"
            )
        child_ids: list[uuid.UUID] = []
        for collection_id in collection_ids:
            child_id = uuid.uuid5(parent.pk, f"collection:{collection_id}")
            child = GraphRebuildRequest.objects.filter(pk=child_id).first()
            if child is None:
                child = _create_scoped_rebuild_request(
                    scope_type=GraphRebuildRequest.ScopeType.COLLECTION,
                    scope_id=collection_id,
                    request_id=child_id,
                    evaluation_only=False,
                    parent_request=parent,
                )
            elif (
                child.parent_request_id != parent.pk
                or child.scope_type != GraphRebuildRequest.ScopeType.COLLECTION
                or child.scope_id != str(collection_id)
            ):
                raise CorruptBuildError("operator rebuild child identity changed")
            child_ids.append(child.pk)
        parent.enumeration_cursor = int(collection_ids[-1])
        parent.expected_child_count += len(child_ids)
        parent.collection_count = parent.expected_child_count
        if parent.enumeration_cursor >= high_water:
            parent.enumeration_complete = True
        parent.save(
            update_fields=[
                "enumeration_cursor",
                "enumeration_complete",
                "expected_child_count",
                "collection_count",
                "updated_at",
            ]
        )
        return tuple(child_ids)


def _request_scope_matches_resume(
    request: object,
    *,
    scope_type: str,
    scope_id: object,
    evaluation_only: bool,
) -> bool:
    if (
        request.scope_type != scope_type
        or request.evaluation_only is not evaluation_only
    ):
        return False
    if scope_type == "all":
        return scope_id is None and request.scope_id == ""
    try:
        expected_scope = (
            str(_canonical_request_uuid(scope_id, required=True))
            if scope_type == "document"
            else (
                str(scope_id) if type(scope_id) is int and 0 < scope_id < 2**63 else ""
            )
        )
    except (ValueError, ValidationError):
        return False
    return request.scope_id == expected_scope


def resume_rebuild_request(request_id: uuid.UUID | str):
    """Idempotently reconcile durable enumeration and at-least-once publication."""

    resolved_id = _canonical_request_uuid(request_id, required=True)
    from apps.knowledge_graph.models import GraphRebuildRequest

    request = GraphRebuildRequest.objects.filter(pk=resolved_id).first()
    if request is None:
        raise LookupError("rebuild request does not exist")
    request = _effective_rebuild_request(request)
    if (
        request.status == GraphRebuildRequest.Status.PARTIAL
        and request.error_code in _RESNAPSHOT_RECONCILABLE_ERRORS
    ):
        _reconcile_rebuild_successor(request.pk)
        request = GraphRebuildRequest.objects.get(pk=resolved_id)
        request = _effective_rebuild_request(request)
    if request.scope_type == GraphRebuildRequest.ScopeType.ALL:
        try:
            _publish_operator_rebuild_children(request.pk)
            while True:
                child_ids = _enumerate_operator_rebuild_page(request.pk)
                if not child_ids:
                    break
                for child_id in child_ids:
                    resume_rebuild_request(child_id)
        except RebuildPublicationError as exc:
            raise RebuildPublicationError(request.pk, exc.error_code) from None
        _advance_parent_rebuild_request(request.pk)
        return GraphRebuildRequest.objects.get(pk=request.pk)
    _publish_rebuild_document_tasks(request.pk)
    request.refresh_from_db()
    if request.status == GraphRebuildRequest.Status.RUNNING:
        advance_rebuild_request(request.pk)
        request.refresh_from_db()
    if (
        request.status == GraphRebuildRequest.Status.RUNNING
        and request.collection_publication_state
        in {
            GraphRebuildRequest.PublicationState.PENDING,
            GraphRebuildRequest.PublicationState.FAILED,
        }
        and request.collection_build_key
        and request.expected_aggregate_signature
    ):
        _publish_correlated_collection_refresh(
            request.pk,
            int(request.scope_id),
            request.expected_aggregate_signature,
            request.collection_build_key,
            request.evaluation_only,
        )
    return GraphRebuildRequest.objects.get(pk=request.pk)


def create_rebuild_request(
    *,
    scope_type: str,
    scope_id: object,
    request_id: uuid.UUID | None = None,
    evaluation_only: bool = False,
    parent_request_id: uuid.UUID | None = None,
):
    """Create or resume one durable request, then reconcile scalar publication."""

    assert_evaluation_bypass(evaluation_only)
    request_uuid = _canonical_request_uuid(request_id) or uuid.uuid4()
    parent_uuid = _canonical_request_uuid(parent_request_id)
    if evaluation_only and scope_type != "collection":
        raise ValueError("evaluation-only rebuild requires one concrete collection")
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphRebuildRequest

    with transaction.atomic():
        _lock_rebuild_request_creation(request_uuid)
        existing = GraphRebuildRequest.objects.filter(pk=request_uuid).first()
        if existing is not None:
            if not _request_scope_matches_resume(
                existing,
                scope_type=scope_type,
                scope_id=scope_id,
                evaluation_only=evaluation_only,
            ):
                raise ValueError(
                    "rebuild request id belongs to another immutable scope"
                )
            if existing.parent_request_id != parent_uuid:
                raise ValueError(
                    "rebuild request id belongs to another immutable parent"
                )
            request = existing
        else:
            parent = None
            if parent_uuid is not None:
                parent = GraphRebuildRequest.objects.select_for_update().get(
                    pk=parent_uuid
                )
            if scope_type in {"document", "collection"}:
                request = _create_scoped_rebuild_request(
                    scope_type=scope_type,
                    scope_id=scope_id,
                    request_id=request_uuid,
                    evaluation_only=evaluation_only,
                    parent_request=parent,
                )
            else:
                if scope_type != "all" or scope_id is not None:
                    raise ValueError("operator-wide rebuild requires an empty scope id")
                if parent is not None:
                    raise ValueError("operator-wide rebuild cannot be a child request")
                high_water = (
                    Collection.objects.order_by("-pk")
                    .values_list("pk", flat=True)
                    .first()
                    or 0
                )
                request = GraphRebuildRequest.objects.create(
                    id=request_uuid,
                    scope_type=GraphRebuildRequest.ScopeType.ALL,
                    scope_id="",
                    requested_documents=[],
                    document_count=0,
                    evaluation_only=False,
                    status=GraphRebuildRequest.Status.RUNNING,
                    started_at=timezone.now(),
                    enumeration_high_water=high_water,
                    document_publication_state=(
                        GraphRebuildRequest.PublicationState.NOT_APPLICABLE
                    ),
                    collection_publication_state=(
                        GraphRebuildRequest.PublicationState.NOT_APPLICABLE
                    ),
                )
    try:
        return resume_rebuild_request(request.pk)
    except RebuildPublicationError as exc:
        raise RebuildPublicationError(request.pk, exc.error_code) from None
    except Exception:
        raise RebuildPublicationError(request.pk, "reconcile_failed") from None


def enqueue_document_build(
    document_id: uuid.UUID,
    expected_source_hash: str,
    request_id: uuid.UUID | str | None = None,
    eval_only: bool = False,
) -> None:
    """Publish one provider-neutral, JSON-safe document build request."""

    from lib.knowledge_graph.config import get_build_enabled

    evaluation_authorized = assert_evaluation_bypass(eval_only)
    resolved_request_id = _canonical_request_uuid(
        request_id, required=evaluation_authorized
    )
    if not get_build_enabled() and not evaluation_authorized:
        if resolved_request_id is not None:
            raise RuntimeError("correlated graph build publication is disabled")
        return
    if type(document_id) is not uuid.UUID:
        raise ValueError("document id must be an exact UUID")
    source_hash = _hash(expected_source_hash, "expected source hash")
    build_key = derive_current_document_build_key(document_id, source_hash)
    from apps.knowledge_graph.tasks import build_document_graph_task

    payload: dict[str, object] = {
        "document_id": str(document_id),
        "expected_source_hash": source_hash,
        "document_build_key": build_key,
    }
    if resolved_request_id is not None or eval_only:
        payload.update(
            {
                "request_id": (
                    None if resolved_request_id is None else str(resolved_request_id)
                ),
                "eval_only": eval_only,
            }
        )
    _publish_graph_task(
        build_document_graph_task,
        kwargs=payload,
        build_kind="document",
        scope_id=str(document_id),
    )
    logger.info(
        "obs.kg.build_stage",
        build_kind="document",
        scope_id=str(document_id),
        stage="build_requested",
        expected_source_hash=source_hash,
        build_key=build_key,
    )


def enqueue_collection_refresh(
    collection_id: int,
    aggregate_source_signature: str,
    collection_build_key: str,
    request_id: uuid.UUID | str | None = None,
    eval_only: bool = False,
) -> None:
    """Publish one exact, JSON-safe collection refresh request."""

    from lib.knowledge_graph.config import get_build_enabled

    evaluation_authorized = assert_evaluation_bypass(eval_only)
    resolved_request_id = _canonical_request_uuid(
        request_id, required=evaluation_authorized
    )
    if not get_build_enabled() and not evaluation_authorized:
        if resolved_request_id is not None:
            raise RuntimeError("correlated graph refresh publication is disabled")
        return
    if type(collection_id) is not int or not 0 < collection_id < 2**63:
        raise ValueError("collection id must be a positive database integer")
    aggregate_signature = _hash(
        aggregate_source_signature,
        "aggregate source signature",
    )
    build_key = _hash(collection_build_key, "collection build key")
    from apps.knowledge_graph.tasks import refresh_collection_graph_task

    payload: dict[str, object] = {
        "collection_id": collection_id,
        "aggregate_source_signature": aggregate_signature,
        "collection_build_key": build_key,
    }
    if resolved_request_id is not None or eval_only:
        payload.update(
            {
                "request_id": (
                    None if resolved_request_id is None else str(resolved_request_id)
                ),
                "eval_only": eval_only,
            }
        )
    _publish_graph_task(
        refresh_collection_graph_task,
        kwargs=payload,
        build_kind="collection",
        scope_id=str(collection_id),
    )

    logger.info(
        "obs.kg.build_stage",
        build_kind="collection",
        scope_id=str(collection_id),
        stage="refresh_requested",
        aggregate_source_signature=aggregate_signature,
        build_key=build_key,
    )


def enqueue_current_collection_refresh(collection_id: int) -> None:
    """Resolve an exact post-commit collection snapshot for the Task 12 seam."""

    try:
        context = _collection_context(collection_id)
        build_key = derive_collection_build_key(context.identity)
    except Exception:
        # Document activation is already durable when this callback runs.  A
        # deleted collection or temporarily unavailable policy snapshot must
        # not turn that committed build into a reported failure.
        logger.error(
            "obs.kg.build_failed",
            build_kind="collection",
            scope_id=str(collection_id),
            stage="refresh_requested",
            error_code="collection_refresh_identity_unavailable",
        )
        return
    enqueue_collection_refresh(
        collection_id,
        context.identity.aggregate_source_signature,
        build_key,
    )


def _document_artifact_values(context: _DocumentContext, build_key: str):
    from apps.knowledge_graph.extraction.pipeline import (
        document_artifact_identity_values,
    )

    values = document_artifact_identity_values(
        context.identity.document_id,
        context.identity.source_hash,
        context.identity.ontology_version,
        context.identity.ontology_checksum,
        settings=context.settings,
    )
    values["build_key"] = build_key
    return values


def _advance_parent_rebuild_request(parent_request_id: uuid.UUID | None) -> None:
    if parent_request_id is None:
        return
    from apps.knowledge_graph.models import GraphRebuildRequest

    if not connection.in_atomic_block:
        with transaction.atomic():
            _advance_parent_rebuild_request(parent_request_id)
        return
    parent = (
        GraphRebuildRequest.objects.select_for_update()
        .filter(pk=parent_request_id)
        .first()
    )
    if parent is None or parent.status != GraphRebuildRequest.Status.RUNNING:
        return
    if (
        parent.scope_type != GraphRebuildRequest.ScopeType.ALL
        or not parent.enumeration_complete
    ):
        return
    terminal = (
        GraphRebuildRequest.Status.SUCCEEDED,
        GraphRebuildRequest.Status.FAILED,
        GraphRebuildRequest.Status.PARTIAL,
    )
    leaves = GraphRebuildRequest.objects.filter(
        Q(parent_request_id=parent.pk) | Q(lineage_root__parent_request_id=parent.pk),
        successor_request__isnull=True,
    )
    outcome = leaves.aggregate(
        total=Count("pk"),
        terminal=Count("pk", filter=Q(status__in=terminal)),
        successes=Count(
            "pk",
            filter=Q(status=GraphRebuildRequest.Status.SUCCEEDED),
        ),
        reconciling=Count(
            "pk",
            filter=Q(error_code__in=_RESNAPSHOT_RECONCILABLE_ERRORS),
        ),
    )
    total = int(outcome["total"] or 0)
    if (
        total != parent.expected_child_count
        or int(outcome["terminal"] or 0) != total
        or int(outcome["reconciling"] or 0) != 0
    ):
        return
    successes = int(outcome["successes"] or 0)
    failures = total - successes
    if failures == 0:
        status = GraphRebuildRequest.Status.SUCCEEDED
    elif successes == 0:
        status = GraphRebuildRequest.Status.FAILED
    else:
        status = GraphRebuildRequest.Status.PARTIAL
    error_code = "" if failures == 0 else "child_rebuild_failed"
    completed_at = timezone.now()
    updated = GraphRebuildRequest.objects.all()._terminalize_operator_parent(
        request_id=parent.pk,
        status=status,
        successes=successes,
        failures=failures,
        expected_children=total,
        error_code=error_code,
        completed_at=completed_at,
    )
    if updated != 1:
        raise CorruptBuildError("operator parent terminal transition was not exact")


def _resnapshot_scope_terminal_error(
    scope_type: str,
    scope_id: str,
) -> str | None:
    """Classify only an exact missing/ineligible scope after a failed lock attempt."""

    from apps.collections.models import Collection
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    from apps.knowledge_graph.models import GraphRebuildRequest

    if scope_type == GraphRebuildRequest.ScopeType.COLLECTION:
        return (
            None
            if Collection._base_manager.filter(pk=int(scope_id)).exists()
            else "scope_deleted"
        )
    if scope_type != GraphRebuildRequest.ScopeType.DOCUMENT:
        return None
    document_id = uuid.UUID(scope_id)
    eligibility = tuple(
        complete
        for model in DESCENDED_FROM_DOCUMENT
        for complete in model._base_manager.filter(id=document_id)
        .values_list("ingestion_complete", flat=True)
        .order_by()[:2]
    )
    if not eligibility:
        return "scope_deleted"
    if eligibility == (False,):
        return "scope_ineligible"
    return None


def _finalize_rebuild_resnapshot_error(
    request_id: uuid.UUID,
    error_code: str,
) -> None:
    """Resolve one provisional terminal marker without reopening request state."""

    from apps.knowledge_graph.models import GraphRebuildRequest

    with transaction.atomic():
        hierarchy_parent, request = _lock_rebuild_request_prefix(request_id)
        if GraphRebuildRequest.objects.filter(
            predecessor_request_id=request.pk
        ).exists():
            return
        if (
            request.status != GraphRebuildRequest.Status.PARTIAL
            or request.error_code not in _RESNAPSHOT_RECONCILABLE_ERRORS
        ):
            return
        if request.error_code != error_code:
            request.error_code = error_code
            request.save(update_fields=["error_code", "updated_at"])
        _advance_parent_rebuild_request(
            None if hierarchy_parent is None else hierarchy_parent.pk
        )


def _create_rebuild_successor_once(
    request_id: uuid.UUID,
) -> tuple[object | None, Exception | None, str, str]:
    """Attempt one successor capture in one self-contained lock transaction."""

    from apps.knowledge_graph.models import GraphRebuildRequest

    with transaction.atomic():
        _hierarchy_parent, request = _lock_rebuild_request_prefix(request_id)
        successor = GraphRebuildRequest.objects.filter(
            predecessor_request_id=request.pk
        ).first()
        if successor is not None:
            return successor, None, request.scope_type, request.scope_id
        if (
            request.status != GraphRebuildRequest.Status.PARTIAL
            or request.error_code not in _RESNAPSHOT_RECONCILABLE_ERRORS
            or request.scope_type
            not in {
                GraphRebuildRequest.ScopeType.DOCUMENT,
                GraphRebuildRequest.ScopeType.COLLECTION,
            }
        ):
            return None, None, request.scope_type, request.scope_id
        scope_id: object = (
            uuid.UUID(request.scope_id)
            if request.scope_type == GraphRebuildRequest.ScopeType.DOCUMENT
            else int(request.scope_id)
        )
        root = request.lineage_root or request
        try:
            successor = _create_scoped_rebuild_request(
                scope_type=request.scope_type,
                scope_id=scope_id,
                request_id=uuid.uuid5(request.pk, "successor"),
                evaluation_only=request.evaluation_only,
                parent_request=None,
                predecessor_request=request,
                lineage_root=root,
            )
        except (LookupError, StaleBuildError) as exc:
            return None, exc, request.scope_type, request.scope_id
        return successor, None, request.scope_type, request.scope_id


def _reconcile_rebuild_successor(request_id: uuid.UUID) -> object | None:
    """Capture and publish a successor with bounded fresh canonical lock attempts."""

    last_failure: Exception | None = None
    for _attempt in range(_MAX_RESNAPSHOT_ATTEMPTS):
        successor, failure, scope_type, scope_id = _create_rebuild_successor_once(
            request_id
        )
        if successor is not None:
            resume_rebuild_request(successor.pk)
            return successor
        if failure is None:
            return None
        last_failure = failure
        terminal_error = _resnapshot_scope_terminal_error(scope_type, scope_id)
        exact_missing_failure = isinstance(failure, LookupError) or (
            isinstance(failure, StaleBuildError)
            and str(failure) == _MISSING_COLLECTION_SNAPSHOT_ERROR
        )
        if terminal_error is not None and exact_missing_failure:
            _finalize_rebuild_resnapshot_error(request_id, terminal_error)
            return None
    if last_failure is not None:
        _finalize_rebuild_resnapshot_error(request_id, _RESNAPSHOT_CHURN_ERROR)
    return None


def _schedule_rebuild_resnapshot(request: object) -> None:
    """Persist a terminal marker, then reconcile only after its locks commit."""

    if request.scope_type not in {
        request.ScopeType.DOCUMENT,
        request.ScopeType.COLLECTION,
    }:
        return
    request.error_code = _RESNAPSHOT_PENDING_ERROR
    transaction.on_commit(
        lambda request_id=request.pk: _reconcile_rebuild_successor(request_id),
        robust=True,
    )


def record_rebuild_failure(
    request_id: uuid.UUID | str | None,
    *,
    error_code: str,
    resnapshot: bool = False,
) -> None:
    """Idempotently terminalize one correlated request without private details."""

    resolved_id = _canonical_request_uuid(request_id)
    if resolved_id is None:
        return
    if (
        type(error_code) is not str
        or not error_code
        or len(error_code) > 128
        or any(value in error_code for value in "\x00\r\n")
    ):
        error_code = "graph_rebuild_failed"
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services.rebuild_failure_locking import (
        locked_completed_document_count,
    )

    with transaction.atomic():
        hierarchy_parent, request = _lock_rebuild_request_prefix(resolved_id)
        if request is None or request.status in {
            GraphRebuildRequest.Status.SUCCEEDED,
            GraphRebuildRequest.Status.FAILED,
            GraphRebuildRequest.Status.PARTIAL,
        }:
            return
        # Fence artifacts NO KEY then runs UPDATE before exact lifecycle reads.
        # This remains compatible with deferred FK checks at worker COMMIT.
        completed = locked_completed_document_count(request)
        request.completed_document_count = min(
            max(request.completed_document_count, completed), request.document_count
        )
        remaining = request.document_count - request.completed_document_count
        if remaining > 0 and request.terminal_failure_count == 0:
            request.terminal_failure_count = 1
        if request.scope_type == GraphRebuildRequest.ScopeType.COLLECTION:
            request.failed_collection_count = 1
        request.status = (
            GraphRebuildRequest.Status.PARTIAL
            if request.completed_document_count or resnapshot
            else GraphRebuildRequest.Status.FAILED
        )
        request.error_code = error_code
        request.completed_at = timezone.now()
        if resnapshot:
            _schedule_rebuild_resnapshot(request)
        request.save(
            update_fields=[
                "completed_document_count",
                "terminal_failure_count",
                "failed_collection_count",
                "status",
                "error_code",
                "completed_at",
                "updated_at",
            ]
        )
        _advance_parent_rebuild_request(
            None if hierarchy_parent is None else hierarchy_parent.pk
        )


_RUN_ARTIFACT_IDENTITY_FIELDS = (
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
)


def _run_identity_matches_artifact(
    artifact: object,
    run: object,
    *,
    build_kind: str,
) -> bool:
    return (
        run is not None
        and getattr(run, "artifact_id", None) == getattr(artifact, "pk", None)
        and getattr(run, "build_kind", None) == build_kind
        and getattr(artifact, "scope_type", None) == build_kind
        and all(
            getattr(run, field, None) == getattr(artifact, field, None)
            for field in _RUN_ARTIFACT_IDENTITY_FIELDS
        )
    )


def _assign_request_activation_audit(
    request: object, artifact: object, run: object
) -> None:
    from apps.knowledge_graph.models.artifacts import _activation_audit_values

    for field, value in _activation_audit_values(artifact, run).items():
        setattr(request, field, value)


def _evaluation_occurrence_completed(
    artifact: object,
    run: object,
    *,
    build_kind: str,
) -> bool:
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    marker = _safe_marker(getattr(run, "stage_marker", None))
    return (
        _run_identity_matches_artifact(
            artifact,
            run,
            build_kind=build_kind,
        )
        and getattr(artifact, "status", None) == GraphArtifact.Status.SUPERSEDED
        and getattr(artifact, "evaluation_only", None) is True
        and getattr(artifact, "rebuild_request_id", None) is not None
        and getattr(run, "stage", None) == GraphBuildRun.Stage.SUPERSEDED
        and getattr(run, "status", None) == GraphBuildRun.Status.CANCELLED
        and marker.get("evaluation_completed") is True
        and getattr(run, "finished_at", None) is not None
        and not getattr(run, "lease_owner", "")
        and getattr(run, "lease_expires_at", None) is None
    )


def _production_occurrence_completed(
    artifact: object,
    run: object,
    *,
    build_kind: str,
    allow_historical: bool,
) -> bool:
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    if not _run_identity_matches_artifact(
        artifact,
        run,
        build_kind=build_kind,
    ):
        return False
    if (
        getattr(artifact, "evaluation_only", None) is not False
        or getattr(artifact, "rebuild_request_id", None) is None
        or getattr(artifact, "activated_at", None) is None
        or getattr(artifact, "completed_at", None) is None
        or getattr(run, "finished_at", None) is None
        or getattr(run, "lease_owner", "")
        or getattr(run, "lease_expires_at", None) is not None
    ):
        return False
    if (
        getattr(artifact, "status", None) == GraphArtifact.Status.ACTIVE
        and getattr(run, "stage", None) == GraphBuildRun.Stage.ACTIVE
        and getattr(run, "status", None) == GraphBuildRun.Status.SUCCEEDED
    ):
        return True
    if not allow_historical:
        return False
    marker = _safe_marker(getattr(run, "stage_marker", None))
    sequence = marker.get("stage_sequence", ())
    return (
        getattr(artifact, "status", None) == GraphArtifact.Status.SUPERSEDED
        and getattr(artifact, "superseded_at", None) is not None
        and getattr(run, "stage", None) == GraphBuildRun.Stage.SUPERSEDED
        and getattr(run, "status", None) == GraphBuildRun.Status.CANCELLED
        and type(sequence) is list
        and GraphBuildRun.Stage.ACTIVE in sequence
        and GraphBuildRun.Stage.SUPERSEDED in sequence
    )


def _completed_request_document_artifacts(
    request: object,
    expected: tuple[dict[str, object], ...],
    *,
    for_update: bool,
) -> tuple[object, ...]:
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    document_ids = tuple(row["document_id"] for row in expected)
    if request.evaluation_only:
        completed_statuses = (GraphArtifact.Status.SUPERSEDED,)
    elif request.scope_type == "document":
        completed_statuses = (
            GraphArtifact.Status.ACTIVE,
            GraphArtifact.Status.SUPERSEDED,
        )
    else:
        completed_statuses = (GraphArtifact.Status.ACTIVE,)
    query = GraphArtifact.objects.filter(
        rebuild_request_id=request.pk,
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id__in=document_ids,
        status__in=completed_statuses,
        evaluation_only=request.evaluation_only,
    ).order_by("scope_id", "pk")
    if for_update:
        query = query.select_for_update()
    artifacts = tuple(query[: len(expected) + 1])
    if len(artifacts) > len(expected):
        raise CorruptBuildError("request owns surplus document artifacts")
    source_by_id = {row["document_id"]: row["source_hash"] for row in expected}
    if any(
        source_by_id.get(artifact.scope_id) != artifact.source_hash
        for artifact in artifacts
    ):
        raise CorruptBuildError("request artifact source differs from its snapshot")
    scope_ids = tuple(artifact.scope_id for artifact in artifacts)
    if len(scope_ids) != len(set(scope_ids)):
        raise CorruptBuildError("request owns duplicate document artifacts")
    if artifacts:
        run_query = GraphBuildRun.objects.filter(
            rebuild_request_id=request.pk,
            artifact_id__in=tuple(artifact.pk for artifact in artifacts),
            build_kind=GraphBuildRun.BuildKind.DOCUMENT,
        ).order_by("artifact_id", "pk")
        if for_update:
            run_query = run_query.select_for_update()
        runs = tuple(run_query[: len(artifacts) + 1])
        run_by_artifact: dict[int, object] = {}
        for run in runs:
            if run.artifact_id in run_by_artifact:
                raise CorruptBuildError("request owns duplicate document build runs")
            run_by_artifact[run.artifact_id] = run
        for artifact in artifacts:
            run = run_by_artifact.get(artifact.pk)
            if request.evaluation_only:
                completed = _evaluation_occurrence_completed(
                    artifact,
                    run,
                    build_kind=GraphBuildRun.BuildKind.DOCUMENT,
                )
            else:
                completed = _production_occurrence_completed(
                    artifact,
                    run,
                    build_kind=GraphBuildRun.BuildKind.DOCUMENT,
                    allow_historical=(request.scope_type == "document"),
                )
            if not completed:
                raise CorruptBuildError(
                    "request document completion lacks its exact terminal run"
                )
    return artifacts


def advance_rebuild_request(request_id: uuid.UUID | str) -> None:
    """Advance a request only from its exact immutable document snapshot."""

    resolved_id = _canonical_request_uuid(request_id, required=True)
    from apps.knowledge_graph.models import (
        GraphBuildRun,
        GraphRebuildRequest,
    )

    with transaction.atomic():
        hierarchy_parent, request = _lock_rebuild_request_prefix(resolved_id)
        hierarchy_parent_id = None if hierarchy_parent is None else hierarchy_parent.pk
        if request.status in {
            GraphRebuildRequest.Status.SUCCEEDED,
            GraphRebuildRequest.Status.FAILED,
            GraphRebuildRequest.Status.PARTIAL,
        }:
            return
        if request.scope_type not in {
            GraphRebuildRequest.ScopeType.DOCUMENT,
            GraphRebuildRequest.ScopeType.COLLECTION,
        }:
            _advance_parent_rebuild_request(request.pk)
            return
        if (
            request.document_publication_state
            != GraphRebuildRequest.PublicationState.PUBLISHED
        ):
            return
        expected = tuple(request.requested_documents)
        try:
            current = _lock_request_completion_snapshot(request, expected)
        except (LookupError, RuntimeError, ValueError):
            current = ()
        if current != expected:
            if request.scope_type == GraphRebuildRequest.ScopeType.COLLECTION:
                request.failed_collection_count = 1
            request.status = GraphRebuildRequest.Status.PARTIAL
            request.error_code = "request_snapshot_changed"
            request.completed_at = timezone.now()
            _schedule_rebuild_resnapshot(request)
            request.save(
                update_fields=[
                    "status",
                    "failed_collection_count",
                    "error_code",
                    "completed_at",
                    "updated_at",
                ]
            )
            _advance_parent_rebuild_request(hierarchy_parent_id)
            return

        active = _completed_request_document_artifacts(
            request,
            expected,
            for_update=True,
        )
        request.completed_document_count = min(len(active), request.document_count)
        failure_query = GraphBuildRun.objects.filter(
            rebuild_request_id=request.pk,
            build_kind=GraphBuildRun.BuildKind.DOCUMENT,
            status__in=(GraphBuildRun.Status.FAILED, GraphBuildRun.Status.CANCELLED),
        )
        if active:
            failure_query = failure_query.exclude(
                artifact_id__in=tuple(artifact.pk for artifact in active)
            )
        terminal_failures = failure_query.count()
        request.terminal_failure_count = min(terminal_failures, request.document_count)
        request.save(
            update_fields=[
                "completed_document_count",
                "terminal_failure_count",
                "updated_at",
            ]
        )
        if terminal_failures:
            if request.scope_type == GraphRebuildRequest.ScopeType.COLLECTION:
                request.failed_collection_count = 1
            request.status = (
                GraphRebuildRequest.Status.PARTIAL
                if active
                else GraphRebuildRequest.Status.FAILED
            )
            request.error_code = "document_rebuild_failed"
            request.completed_at = timezone.now()
            request.save(
                update_fields=[
                    "status",
                    "failed_collection_count",
                    "error_code",
                    "completed_at",
                    "updated_at",
                ]
            )
            _advance_parent_rebuild_request(hierarchy_parent_id)
            return
        if len(active) != len(expected):
            return
        if request.scope_type == GraphRebuildRequest.ScopeType.DOCUMENT:
            artifact = active[0]
            runs = tuple(
                GraphBuildRun.objects.select_for_update()
                .filter(
                    artifact_id=artifact.pk,
                    rebuild_request_id=request.pk,
                    build_kind=GraphBuildRun.BuildKind.DOCUMENT,
                )
                .order_by("pk")[:2]
            )
            if len(runs) != 1:
                raise CorruptBuildError(
                    "document activation lacks one exact terminal run"
                )
            request.status = GraphRebuildRequest.Status.SUCCEEDED
            _assign_request_activation_audit(request, artifact, runs[0])
            request.completed_at = timezone.now()
            request.save(
                update_fields=[
                    "status",
                    "activated_artifact_pk",
                    "activated_run_pk",
                    "activated_build_key",
                    "activated_build_generation",
                    "activated_source_hash",
                    "activated_occurrence_signature",
                    "completed_at",
                    "updated_at",
                ]
            )
            collection_id = int(expected[0]["collection_id"])
            transaction.on_commit(
                lambda: enqueue_current_collection_refresh(collection_id),
                robust=True,
            )
            _advance_parent_rebuild_request(hierarchy_parent_id)
            return
        if request.collection_refresh_enqueued_at is not None:
            return
        try:
            context = _collection_context(
                int(request.scope_id),
                document_artifacts_override=(
                    active if request.evaluation_only else None
                ),
                evaluation_request_id=(request.pk if request.evaluation_only else None),
            )
        except Exception:
            request.failed_collection_count = 1
            request.status = GraphRebuildRequest.Status.PARTIAL
            request.error_code = "collection_manifest_changed"
            request.completed_at = timezone.now()
            _schedule_rebuild_resnapshot(request)
            request.save(
                update_fields=[
                    "status",
                    "failed_collection_count",
                    "error_code",
                    "completed_at",
                    "updated_at",
                ]
            )
            _advance_parent_rebuild_request(hierarchy_parent_id)
            return
        contributing_ids = {artifact.pk for artifact in context.document_artifacts}
        if contributing_ids != {artifact.pk for artifact in active}:
            request.failed_collection_count = 1
            request.status = GraphRebuildRequest.Status.PARTIAL
            request.error_code = "collection_manifest_changed"
            request.completed_at = timezone.now()
            _schedule_rebuild_resnapshot(request)
            request.save(
                update_fields=[
                    "status",
                    "failed_collection_count",
                    "error_code",
                    "completed_at",
                    "updated_at",
                ]
            )
            _advance_parent_rebuild_request(hierarchy_parent_id)
            return
        build_key = derive_collection_build_key(context.identity)
        request.expected_aggregate_signature = (
            context.identity.aggregate_source_signature
        )
        request.collection_build_key = build_key
        request.collection_refresh_enqueued_at = timezone.now()
        request.collection_publication_state = (
            GraphRebuildRequest.PublicationState.PENDING
        )
        request.error_code = ""
        request.save(
            update_fields=[
                "expected_aggregate_signature",
                "collection_build_key",
                "collection_refresh_enqueued_at",
                "collection_publication_state",
                "error_code",
                "updated_at",
            ]
        )
        transaction.on_commit(
            lambda: _publish_correlated_collection_refresh(
                request.pk,
                context.identity.collection_id,
                context.identity.aggregate_source_signature,
                build_key,
                request.evaluation_only,
            ),
            robust=False,
        )


def _publish_correlated_collection_refresh(
    request_id: uuid.UUID,
    collection_id: int,
    aggregate_source_signature: str,
    collection_build_key: str,
    eval_only: bool,
) -> None:
    """Publish post-commit and leave a retryable durable marker on failure."""

    try:
        enqueue_collection_refresh(
            collection_id,
            aggregate_source_signature,
            collection_build_key,
            request_id=request_id,
            eval_only=eval_only,
        )
    except Exception:
        from apps.knowledge_graph.models import GraphRebuildRequest

        with transaction.atomic():
            _parent, request = _lock_rebuild_request_prefix(request_id)
            if (
                request.status == GraphRebuildRequest.Status.RUNNING
                and request.activated_artifact_pk is None
            ):
                request.collection_publication_state = (
                    GraphRebuildRequest.PublicationState.FAILED
                )
                request.error_code = "collection_refresh_publish_failed"
                request.save(
                    update_fields=[
                        "collection_publication_state",
                        "error_code",
                        "updated_at",
                    ]
                )
        raise RebuildPublicationError(
            request_id,
            "collection_refresh_publish_failed",
        ) from None
    from apps.knowledge_graph.models import GraphRebuildRequest

    with transaction.atomic():
        _parent, request = _lock_rebuild_request_prefix(request_id)
        if request.status == GraphRebuildRequest.Status.RUNNING:
            request.collection_publication_state = (
                GraphRebuildRequest.PublicationState.PUBLISHED
            )
            request.collection_refresh_published_at = timezone.now()
            if request.error_code == "collection_refresh_publish_failed":
                request.error_code = ""
            request.save(
                update_fields=[
                    "collection_publication_state",
                    "collection_refresh_published_at",
                    "error_code",
                    "updated_at",
                ]
            )
    from apps.knowledge_graph.models import GraphArtifact

    completed_artifact = (
        GraphArtifact.objects.filter(
            rebuild_request_id=request_id,
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            status__in=(
                GraphArtifact.Status.ACTIVE,
                GraphArtifact.Status.SUPERSEDED,
            ),
        )
        .order_by("-build_generation", "-pk")
        .first()
    )
    if completed_artifact is not None:
        complete_collection_rebuild(request_id, completed_artifact)


def complete_collection_rebuild(
    request_id: uuid.UUID | str | None, artifact: object
) -> None:
    resolved_id = _canonical_request_uuid(request_id)
    if resolved_id is None:
        return
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )

    with transaction.atomic():
        hierarchy_parent, request = _lock_rebuild_request_prefix(resolved_id)
        if request.status == GraphRebuildRequest.Status.SUCCEEDED:
            return
        if (
            request.collection_publication_state
            != GraphRebuildRequest.PublicationState.PUBLISHED
        ):
            return
        artifact_pk = getattr(artifact, "pk", None)
        if type(artifact_pk) is not int or artifact_pk <= 0:
            raise CorruptBuildError("collection completion artifact id is invalid")
        correlated_artifacts = tuple(
            GraphArtifact.objects.select_for_update()
            .filter(
                pk=artifact_pk,
                rebuild_request_id=request.pk,
                scope_type=GraphArtifact.ScopeType.COLLECTION,
                scope_id=request.scope_id,
                evaluation_only=request.evaluation_only,
                source_hash=request.expected_aggregate_signature,
            )
            .order_by("pk")[:2]
        )
        if len(correlated_artifacts) != 1:
            raise CorruptBuildError(
                "collection activation does not match its rebuild request"
            )
        artifact = correlated_artifacts[0]
        if request.evaluation_only:
            artifact_status_valid = artifact.status == GraphArtifact.Status.SUPERSEDED
        else:
            artifact_status_valid = (
                artifact.status
                in {
                    GraphArtifact.Status.ACTIVE,
                    GraphArtifact.Status.SUPERSEDED,
                }
                and artifact.activated_at is not None
            )
        if (
            request.scope_type != GraphRebuildRequest.ScopeType.COLLECTION
            or not artifact_status_valid
        ):
            raise CorruptBuildError(
                "collection activation does not match its rebuild request"
            )
        runs = tuple(
            GraphBuildRun.objects.select_for_update()
            .filter(
                artifact_id=artifact.pk,
                rebuild_request_id=request.pk,
                build_kind=GraphBuildRun.BuildKind.COLLECTION,
            )
            .order_by("pk")[:2]
        )
        if len(runs) != 1:
            raise CorruptBuildError(
                "collection completion lacks one exact terminal run"
            )
        if request.evaluation_only:
            completed = _evaluation_occurrence_completed(
                artifact,
                runs[0],
                build_kind=GraphBuildRun.BuildKind.COLLECTION,
            )
        else:
            completed = _production_occurrence_completed(
                artifact,
                runs[0],
                build_kind=GraphBuildRun.BuildKind.COLLECTION,
                allow_historical=True,
            )
        if not completed:
            raise CorruptBuildError(
                "collection completion lacks its exact terminal run"
            )
        request.status = GraphRebuildRequest.Status.SUCCEEDED
        _assign_request_activation_audit(request, artifact, runs[0])
        request.completed_collection_count = 1
        request.completed_at = timezone.now()
        request.save(
            update_fields=[
                "status",
                "activated_artifact_pk",
                "activated_run_pk",
                "activated_build_key",
                "activated_build_generation",
                "activated_source_hash",
                "activated_occurrence_signature",
                "completed_collection_count",
                "completed_at",
                "updated_at",
            ]
        )
        _advance_parent_rebuild_request(
            None if hierarchy_parent is None else hierarchy_parent.pk
        )


def _register_document_refresh_callbacks(
    context: _DocumentContext,
    run: object,
) -> None:
    rebuild_request_id = getattr(run, "rebuild_request_id", None)
    if rebuild_request_id is not None:
        transaction.on_commit(
            lambda: advance_rebuild_request(rebuild_request_id),
            robust=False,
        )
        return
    metadata = run.metadata if type(run.metadata) is dict else {}
    initial_collection_id = metadata.get("initial_collection_id")
    affected = {context.collection_id}
    if type(initial_collection_id) is int and initial_collection_id > 0:
        affected.add(initial_collection_id)
    for collection_id in sorted(affected):
        transaction.on_commit(
            lambda collection_id=collection_id: enqueue_current_collection_refresh(
                collection_id
            ),
            robust=True,
        )


def _bootstrap_document_build(
    context: _DocumentContext,
    build_key: str,
    request_id: uuid.UUID | None = None,
    eval_only: bool = False,
) -> tuple[object, object, str | None, int | None, bool]:
    from apps.knowledge_graph.extraction.pipeline import (
        _validate_source,
        resolve_ontology_definition,
    )
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )
    from lib.knowledge_graph.config import load_extraction_settings

    owner = uuid.uuid4().hex
    with transaction.atomic():
        request = None
        if request_id is not None:
            try:
                _parent, request = _lock_rebuild_request_prefix(request_id)
            except LookupError as exc:
                raise StaleBuildError("rebuild request no longer exists") from exc
        artifacts, runs, document, chunks = _lock_document_build_rows(
            context.identity.document_id,
            build_key=build_key,
        )
        scope_runs = _lock_latest_scope_run(
            GraphBuildRun.BuildKind.DOCUMENT,
            context.identity.document_id,
        )
        try:
            _validate_source(document, context.identity.source_hash)
        except Exception as exc:
            raise StaleBuildError("document changed before build bootstrap") from exc
        locked_ontology = resolve_ontology_definition(
            context.identity.ontology_version,
            for_update=True,
        )
        current_settings = load_extraction_settings()
        locked_context = _document_context(
            context.identity.document_id,
            context.identity.source_hash,
            ontology=locked_ontology,
            settings=current_settings,
        )
        if locked_context.identity != context.identity:
            raise StaleBuildError("document build identity changed before bootstrap")
        if derive_document_build_key(locked_context.identity) != build_key:
            raise StaleBuildError("document build key changed before bootstrap")

        if request_id is not None:
            assert request is not None
            if request.evaluation_only is not eval_only:
                raise CorruptBuildError("rebuild evaluation marker changed")
            matching = tuple(
                row
                for row in request.requested_documents
                if row.get("document_id") == str(context.identity.document_id)
                and row.get("source_hash") == context.identity.source_hash
            )
            if len(matching) != 1:
                raise StaleBuildError("document is outside the rebuild snapshot")
            correlated = tuple(
                GraphArtifact.objects.select_for_update()
                .filter(
                    rebuild_request_id=request.pk,
                    scope_type=GraphArtifact.ScopeType.DOCUMENT,
                    scope_id=str(context.identity.document_id),
                )
                .order_by("pk")[:2]
            )
            if len(correlated) > 1:
                raise CorruptBuildError("request owns duplicate document occurrences")
            for row in correlated:
                if all(existing.pk != row.pk for existing in artifacts):
                    artifacts = (*artifacts, row)
            correlated_run_rows = tuple(
                GraphBuildRun.objects.select_for_update()
                .filter(
                    rebuild_request_id=request.pk,
                    build_kind=GraphBuildRun.BuildKind.DOCUMENT,
                    scope_id=str(context.identity.document_id),
                )
                .order_by("pk")[:2]
            )
            if len(correlated_run_rows) > 1:
                raise CorruptBuildError("request owns duplicate document build runs")
            for row in correlated_run_rows:
                if all(existing.pk != row.pk for existing in runs):
                    runs = (*runs, row)
            if request.status in {
                GraphRebuildRequest.Status.SUCCEEDED,
                GraphRebuildRequest.Status.FAILED,
                GraphRebuildRequest.Status.PARTIAL,
            }:
                if correlated and correlated_run_rows:
                    return correlated[0], correlated_run_rows[0], None, None, True
                raise StaleBuildError("rebuild request is already terminal")
            if correlated and correlated[0].status == GraphArtifact.Status.SUPERSEDED:
                completed_run = (
                    correlated_run_rows[0] if len(correlated_run_rows) == 1 else None
                )
                if (
                    request.evaluation_only
                    and _evaluation_occurrence_completed(
                        correlated[0],
                        completed_run,
                        build_kind=GraphBuildRun.BuildKind.DOCUMENT,
                    )
                ) or (
                    not request.evaluation_only
                    and request.scope_type == GraphRebuildRequest.ScopeType.DOCUMENT
                    and _production_occurrence_completed(
                        correlated[0],
                        completed_run,
                        build_kind=GraphBuildRun.BuildKind.DOCUMENT,
                        allow_historical=True,
                    )
                ):
                    _register_document_refresh_callbacks(
                        locked_context,
                        completed_run,
                    )
                    return (
                        correlated[0],
                        correlated_run_rows[0],
                        None,
                        None,
                        True,
                    )
                raise StaleBuildError("request document occurrence was superseded")
            action = (
                _occurrence_action(correlated, correlated_run_rows, build_key)
                if correlated
                else OccurrenceAction.CREATE
            )
            action_artifacts = correlated
            action_runs = correlated_run_rows
        else:
            action_artifacts = tuple(
                row for row in artifacts if row.evaluation_only is False
            )
            action_artifact_ids = {row.pk for row in action_artifacts}
            action_runs = tuple(
                row
                for row in runs
                if row.evaluation_only is False
                and row.artifact_id in action_artifact_ids
            )
            action = _occurrence_action(action_artifacts, action_runs, build_key)
        run_by_artifact = {row.artifact_id: row for row in action_runs}
        artifact = None
        run = None
        if action is OccurrenceAction.RETURN_ACTIVE:
            artifact = next(
                row
                for row in action_artifacts
                if row.build_key == build_key
                and row.status == GraphArtifact.Status.ACTIVE
                and row.orchestration_version
                == GraphArtifact.OrchestrationVersion.SCOPED_V1
            )
            run = run_by_artifact[artifact.pk]
            if run.lease_owner or run.lease_expires_at is not None:
                raise CorruptBuildError("active document owns a build lease")
            _register_document_refresh_callbacks(context, run)
            return artifact, run, None, None, True
        if action in {OccurrenceAction.RESUME, OccurrenceAction.RETRY}:
            artifact = max(
                (
                    row
                    for row in action_artifacts
                    if row.build_key == build_key
                    and row.orchestration_version
                    == GraphArtifact.OrchestrationVersion.SCOPED_V1
                ),
                key=lambda row: (row.build_generation, row.pk),
            )
            run = run_by_artifact[artifact.pk]
            _validate_retryable_run(run)
            if _run_has_live_lease(run):
                raise BuildInProgressError(
                    "exact document graph build already has a live lease"
                )
            _restart_locked_run(run)
            if artifact.status in {
                GraphArtifact.Status.FAILED,
                GraphArtifact.Status.STALE,
            }:
                artifact.status = GraphArtifact.Status.BUILDING
                artifact.save(update_fields=["status"])
            elif artifact.status != GraphArtifact.Status.BUILDING:
                raise CorruptBuildError("document retry artifact is not reusable")
        else:
            build_generation = _next_build_generation(artifacts, scope_runs)
            artifact = GraphArtifact.objects.create(
                status=GraphArtifact.Status.BUILDING,
                rebuild_request=request,
                evaluation_only=eval_only,
                build_generation=build_generation,
                orchestration_version=(GraphArtifact.OrchestrationVersion.SCOPED_V1),
                metadata={
                    "orchestration_version": 1,
                    "ordered_chunk_signature": (
                        context.identity.ordered_chunk_signature
                    ),
                    "ontology_activation_signature": (
                        context.identity.ontology_activation_signature
                    ),
                },
                **_document_artifact_values(context, build_key),
            )
            run = GraphBuildRun.objects.create(
                artifact=artifact,
                rebuild_request=request,
                evaluation_only=eval_only,
                build_generation=build_generation,
                orchestration_version=(GraphArtifact.OrchestrationVersion.SCOPED_V1),
                stage=GraphBuildRun.Stage.QUEUED,
                status=GraphBuildRun.Status.PENDING,
                attempt=1,
                metadata={
                    "orchestration_version": 1,
                    "initial_collection_id": context.collection_id,
                    "attempt_history": [],
                },
                stage_marker={
                    "orchestration_version": 1,
                    "build_key": build_key,
                    "ordered_chunk_signature": (
                        context.identity.ordered_chunk_signature
                    ),
                    "stage_sequence": [GraphBuildRun.Stage.QUEUED],
                    "last_stage": GraphBuildRun.Stage.QUEUED,
                },
            )
        lease_owner, lease_generation = _claim_locked_run(run, owner)
        return artifact, run, lease_owner, lease_generation, False


def _document_commit_counts(artifact: object, run: object) -> dict[str, int]:
    from apps.knowledge_graph.models import (
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        RelationMention,
    )

    mention_count = EntityMention.objects.filter(artifact=artifact).count()
    relation_count = RelationMention.objects.filter(artifact=artifact).count()
    if _document_extraction_commit_state(artifact, run) is not CommitMarkerState.VALID:
        raise CorruptBuildError("document extraction commit is incomplete")
    entity_count = DocumentEntity.objects.filter(artifact=artifact).count()
    membership_count = DocumentEntityMention.objects.filter(
        document_entity__artifact=artifact
    ).count()
    if _document_resolution_commit_state(artifact, run) is not CommitMarkerState.VALID:
        raise CorruptBuildError("document resolution commit is incomplete")
    return {
        "entity_mention_count": mention_count,
        "relation_mention_count": relation_count,
        "document_entity_count": entity_count,
        "membership_count": membership_count,
    }


def _apply_locked_terminal(
    run: object,
    target: str,
    *,
    error_code: str = "",
) -> None:
    from apps.knowledge_graph.models import GraphBuildRun

    validate_stage_transition(run.build_kind, run.stage, target)
    run.stage = target
    run.status = {
        GraphBuildRun.Stage.ACTIVE: GraphBuildRun.Status.SUCCEEDED,
        GraphBuildRun.Stage.FAILED: GraphBuildRun.Status.FAILED,
        GraphBuildRun.Stage.SUPERSEDED: GraphBuildRun.Status.CANCELLED,
        GraphBuildRun.Stage.STALE: GraphBuildRun.Status.CANCELLED,
    }[target]
    run.error_code = error_code
    run.error_message = error_code
    run.finished_at = timezone.now()
    run.lease_owner = ""
    run.lease_expires_at = None
    marker = _safe_marker(run.stage_marker)
    sequence = marker.get("stage_sequence", [])
    sequence = list(sequence[-31:]) if type(sequence) is list else []
    sequence.append(target)
    marker["stage_sequence"] = sequence[-32:]
    marker["last_stage"] = target
    run.stage_marker = marker
    run.save(
        update_fields=[
            "stage",
            "status",
            "error_code",
            "error_message",
            "finished_at",
            "lease_owner",
            "lease_expires_at",
            "stage_marker",
        ]
    )


def _complete_evaluation_document_occurrence(
    artifact: object,
    run: object,
) -> None:
    """Terminalize a private evaluation occurrence without exposing it as current."""

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    if (
        getattr(artifact, "evaluation_only", None) is not True
        or getattr(run, "evaluation_only", None) is not True
        or getattr(artifact, "rebuild_request_id", None) is None
        or getattr(run, "rebuild_request_id", None)
        != getattr(artifact, "rebuild_request_id", None)
        or getattr(artifact, "status", None) != GraphArtifact.Status.BUILDING
        or getattr(run, "stage", None) != GraphBuildRun.Stage.VALIDATING
        or getattr(run, "status", None) != GraphBuildRun.Status.RUNNING
    ):
        raise CorruptBuildError(
            "evaluation completion does not own an exact validating occurrence"
        )
    completed_at = timezone.now()
    artifact.status = GraphArtifact.Status.SUPERSEDED
    artifact.completed_at = completed_at
    artifact.superseded_at = completed_at
    artifact.save(update_fields=["status", "completed_at", "superseded_at"])
    marker = _safe_marker(run.stage_marker)
    sequence = marker.get("stage_sequence", [])
    sequence = list(sequence[-31:]) if type(sequence) is list else []
    sequence.append(GraphBuildRun.Stage.SUPERSEDED)
    marker["stage_sequence"] = sequence[-32:]
    marker["last_stage"] = GraphBuildRun.Stage.SUPERSEDED
    marker["evaluation_completed"] = True
    run.stage_marker = marker
    run.stage = GraphBuildRun.Stage.SUPERSEDED
    run.status = GraphBuildRun.Status.CANCELLED
    run.finished_at = completed_at
    run.lease_owner = ""
    run.lease_expires_at = None
    run.save(
        update_fields=[
            "stage",
            "status",
            "finished_at",
            "lease_owner",
            "lease_expires_at",
            "stage_marker",
        ]
    )


def _activate_document_build(
    context: _DocumentContext,
    artifact_id: int,
    run_id: int,
    *,
    lease_owner: str,
    lease_generation: int,
):
    from apps.knowledge_graph.extraction.pipeline import (
        _validate_source,
        resolve_ontology_definition,
    )
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from lib.knowledge_graph.config import load_extraction_settings

    with transaction.atomic():
        artifacts, runs, document, chunks = _lock_document_build_rows(
            context.identity.document_id,
            build_key=derive_document_build_key(context.identity),
            candidate_artifact_id=artifact_id,
        )
        artifact = next((row for row in artifacts if row.pk == artifact_id), None)
        run = next((row for row in runs if row.pk == run_id), None)
        if artifact is None or run is None or run.artifact_id != artifact.pk:
            raise CorruptBuildError("document activation ownership changed")
        validate_build_lease(run, lease_owner, lease_generation)
        try:
            _validate_source(document, context.identity.source_hash)
        except Exception as exc:
            raise StaleBuildError("document changed before activation") from exc
        locked_ontology = resolve_ontology_definition(
            context.identity.ontology_version,
            for_update=True,
        )
        current_context = _document_context(
            context.identity.document_id,
            context.identity.source_hash,
            ontology=locked_ontology,
            settings=load_extraction_settings(),
        )
        current_key = derive_document_build_key(current_context.identity)
        if (
            current_context.identity != context.identity
            or current_key != artifact.build_key
        ):
            raise StaleBuildError("document build identity changed before activation")
        if (
            ordered_chunk_signature(
                chunks,
                concrete_model_label=document._meta.label_lower,
            )
            != context.identity.ordered_chunk_signature
        ):
            raise StaleBuildError("document chunks changed before activation")
        if (
            run.build_key != artifact.build_key
            or run.build_generation != artifact.build_generation
            or run.orchestration_version != artifact.orchestration_version
            or run.stage != GraphBuildRun.Stage.VALIDATING
        ):
            raise CorruptBuildError(
                "document candidate is not validating its exact occurrence"
            )
        counts = _document_commit_counts(artifact, run)
        active = tuple(
            row
            for row in artifacts
            if row.status == GraphArtifact.Status.ACTIVE
            and not row.evaluation_only
            and row.pk != artifact.pk
        )
        if len(active) > 1:
            raise CorruptBuildError("document scope has multiple active artifacts")
        higher_current = tuple(
            row
            for row in artifacts
            if row.build_generation > artifact.build_generation
            and row.status
            in {GraphArtifact.Status.BUILDING, GraphArtifact.Status.ACTIVE}
            and row.evaluation_only is artifact.evaluation_only
            and (
                not artifact.evaluation_only
                or row.rebuild_request_id == artifact.rebuild_request_id
            )
        )
        if higher_current:
            artifact.status = GraphArtifact.Status.STALE
            artifact.completed_at = timezone.now()
            artifact.save(update_fields=["status", "completed_at"])
            _apply_locked_terminal(
                run, GraphBuildRun.Stage.STALE, error_code="newer_document_build_won"
            )

            def report_newer_winner() -> None:
                raise StaleBuildError(
                    "newer document graph artifact already won activation"
                )

            transaction.on_commit(report_newer_winner)
            return max(
                higher_current,
                key=lambda row: (row.build_generation, row.pk),
            ), counts
        if artifact.evaluation_only:
            _complete_evaluation_document_occurrence(artifact, run)
            _register_document_refresh_callbacks(current_context, run)
            return artifact, counts
        now = timezone.now()
        for previous in active:
            previous.status = GraphArtifact.Status.SUPERSEDED
            if previous.superseded_at is None:
                previous.superseded_at = now
            previous.save(update_fields=["status", "superseded_at"])
            previous_run = next(
                (
                    candidate
                    for candidate in reversed(runs)
                    if candidate.artifact_id == previous.pk
                    and candidate.stage == GraphBuildRun.Stage.ACTIVE
                ),
                None,
            )
            if previous_run is not None:
                _apply_locked_terminal(previous_run, GraphBuildRun.Stage.SUPERSEDED)
        artifact.status = GraphArtifact.Status.ACTIVE
        artifact.activated_at = now
        artifact.completed_at = now
        artifact.save(update_fields=["status", "activated_at", "completed_at"])
        _apply_locked_terminal(run, GraphBuildRun.Stage.ACTIVE)
        _register_document_refresh_callbacks(current_context, run)
        return artifact, counts


def _terminal_document_build(
    context: _DocumentContext,
    artifact_id: int,
    run_id: int,
    *,
    lease_owner: str,
    lease_generation: int,
    stale: bool,
    error_code: str,
) -> None:
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    with transaction.atomic():
        artifact, run = _lock_terminal_document_rows(
            context.identity.document_id,
            artifact_id,
            run_id,
        )
        if artifact is None or run is None:
            return
        validate_build_lease(run, lease_owner, lease_generation)
        target = GraphBuildRun.Stage.STALE if stale else GraphBuildRun.Stage.FAILED
        artifact.status = (
            GraphArtifact.Status.STALE if stale else GraphArtifact.Status.FAILED
        )
        artifact.completed_at = timezone.now()
        artifact.save(update_fields=["status", "completed_at"])
        _apply_locked_terminal(run, target, error_code=error_code)


def build_document_graph(
    document_id,
    expected_source_hash,
    document_build_key,
    request_id=None,
    eval_only=False,
):
    """Build and atomically activate one exact document graph."""

    request = _validated_rebuild_request_before_content(
        request_id,
        eval_only,
        build_kind="document",
        scope_id=document_id,
        source_hash=expected_source_hash,
    )
    resolved_request_id = None if request is None else request.pk
    from apps.knowledge_graph.extraction.pipeline import (
        StaleSourceError,
        extract_into_build,
    )
    from apps.knowledge_graph.models import EntityMention, GraphBuildRun
    from apps.knowledge_graph.resolution.coreference import (
        MAX_DOCUMENT_MENTIONS,
        resolve_document_mentions,
    )
    from apps.knowledge_graph.resolution.persistence import (
        _bounded_rows,
        persist_document_resolution,
    )

    started = perf_counter()
    context = _document_context(document_id, expected_source_hash)
    requested_key = _hash(document_build_key, "document build key")
    if derive_document_build_key(context.identity) != requested_key:
        raise StaleBuildError("document build key does not match live source")
    if resolved_request_id is None and not eval_only:
        artifact, run, lease_owner, lease_generation, completed = (
            _bootstrap_document_build(context, requested_key)
        )
    else:
        artifact, run, lease_owner, lease_generation, completed = (
            _bootstrap_document_build(
                context,
                requested_key,
                resolved_request_id,
                eval_only,
            )
        )
    if completed:
        return artifact
    assert lease_owner is not None and lease_generation is not None
    logger.info(
        "obs.kg.build_started",
        build_kind="document",
        scope_id=str(context.identity.document_id),
        build_key=requested_key,
        artifact_id=artifact.pk,
        build_run_id=run.pk,
        attempt=run.attempt,
        ontology_version=context.identity.ontology_version,
        resolver_version=context.identity.resolver_version,
    )
    try:
        if run.stage == GraphBuildRun.Stage.QUEUED:
            run = _transition_run(
                run.pk,
                GraphBuildRun.Stage.EXTRACTING,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
            )
        if run.stage == GraphBuildRun.Stage.EXTRACTING:
            extraction_state = _document_extraction_commit_state(artifact, run)
            if extraction_state is CommitMarkerState.CORRUPT:
                raise CorruptBuildError("document extraction commit is corrupt")
            if extraction_state is CommitMarkerState.ABSENT:
                with LeaseHeartbeat(run.pk, lease_owner, lease_generation):
                    extract_into_build(
                        artifact.pk,
                        run.pk,
                        context.identity.document_id,
                        context.identity.source_hash,
                        context.identity.ontology_version,
                        lease_owner=lease_owner,
                        lease_generation=lease_generation,
                    )
            run = _transition_run(
                run.pk,
                GraphBuildRun.Stage.RESOLVING,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
            )
        if run.stage == GraphBuildRun.Stage.RESOLVING:
            resolution_state = _document_resolution_commit_state(artifact, run)
            if resolution_state is CommitMarkerState.CORRUPT:
                raise CorruptBuildError("document resolution commit is corrupt")
            if resolution_state is CommitMarkerState.ABSENT:
                with LeaseHeartbeat(run.pk, lease_owner, lease_generation):
                    mention_query = EntityMention.objects.filter(
                        artifact=artifact
                    ).order_by("pk")
                    try:
                        mentions = _bounded_rows(
                            mention_query,
                            MAX_DOCUMENT_MENTIONS,
                            "document mention",
                        )
                    except ValueError as exc:
                        raise CorruptBuildError(str(exc)) from exc
                    result = resolve_document_mentions(mentions, context.ontology)
                    persist_document_resolution(
                        artifact.pk,
                        run.pk,
                        result,
                        lease_owner=lease_owner,
                        lease_generation=lease_generation,
                    )
            run = _transition_run(
                run.pk,
                GraphBuildRun.Stage.VALIDATING,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
            )
        activated, counts = _activate_document_build(
            context,
            artifact.pk,
            run.pk,
            lease_owner=lease_owner,
            lease_generation=lease_generation,
        )
        logger.info(
            "obs.kg.build_completed",
            build_kind="document",
            scope_id=str(context.identity.document_id),
            build_key=requested_key,
            artifact_id=activated.pk,
            build_run_id=run.pk,
            attempt=run.attempt,
            total_seconds=perf_counter() - started,
            **counts,
        )
        return activated
    except Exception as exc:
        stale = isinstance(exc, (StaleBuildError, StaleSourceError))
        error_code = (
            "source_or_config_stale"
            if stale
            else (
                "corrupt_build_state"
                if isinstance(exc, CorruptBuildError)
                else "document_build_failed"
            )
        )
        try:
            _terminal_document_build(
                context,
                artifact.pk,
                run.pk,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
                stale=stale,
                error_code=error_code,
            )
        except Exception:
            logger.error(
                "obs.kg.build_terminal_failed",
                build_kind="document",
                scope_id=str(context.identity.document_id),
                build_key=requested_key,
                artifact_id=artifact.pk,
                build_run_id=run.pk,
                attempt=run.attempt,
                error_code="terminal_bookkeeping_failed",
            )
        if stale or isinstance(exc, CorruptBuildError):
            record_rebuild_failure(
                resolved_request_id,
                error_code=error_code,
                resnapshot=stale,
            )
        logger.error(
            "obs.kg.build_failed",
            build_kind="document",
            scope_id=str(context.identity.document_id),
            build_key=requested_key,
            artifact_id=artifact.pk,
            build_run_id=run.pk,
            attempt=run.attempt,
            stage=run.stage,
            error_code=error_code,
            total_seconds=perf_counter() - started,
        )
        raise


@dataclass(frozen=True, slots=True)
class _CollectionContext:
    identity: CollectionBuildIdentity
    collection: object
    document_artifacts: tuple[object, ...]
    ontology: object
    filter_policy: object
    resolution_config: object
    assembly_config: object


def _collection_extractor_version(artifacts: tuple[object, ...]) -> str:
    versions = tuple(sorted({artifact.extractor_version for artifact in artifacts}))
    if not versions:
        return "empty-manifest-v1"
    if len(versions) == 1:
        return _text(
            versions[0],
            "collection extractor version",
            maximum=128,
        )
    return f"manifest-extractors-v1:{_identity_key('extractors-v1', versions)}"


def _validate_collection_context_caps(
    *,
    document_count: int,
    entity_count: int,
    resolution_config: object,
    assembly_config: object,
) -> None:
    """Reject oversized graph inputs without coupling them to raw source volume."""

    counts = {
        "document": document_count,
        "entity": entity_count,
    }
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise CorruptBuildError(
            "collection context counts must be nonnegative integers"
        )
    document_cap = min(
        resolution_config.max_document_inputs,
        assembly_config.max_document_inputs,
    )
    entity_cap = min(resolution_config.max_entities, assembly_config.max_entities)
    for name, value, cap in (
        ("document", document_count, document_cap),
        ("entity", entity_count, entity_cap),
    ):
        if value > cap:
            raise CorruptBuildError(f"collection context {name} cap exceeded")


def _bounded_context_rows(values, maximum: int, label: str) -> tuple[object, ...]:
    """Bound both the preflight count and the post-count iterator snapshot."""

    if type(maximum) is not int or maximum < 1:
        raise CorruptBuildError(f"{label} cap is invalid")
    count = values.count()
    if type(count) is not int or count < 0:
        raise CorruptBuildError(f"{label} count is invalid")
    if count > maximum:
        raise CorruptBuildError(f"{label} cap exceeded")
    rows = tuple(
        islice(
            values.iterator(chunk_size=min(maximum, 1_000)),
            maximum + 1,
        )
    )
    if len(rows) > maximum:
        raise CorruptBuildError(f"{label} cap exceeded")
    return rows


def _collection_context(
    collection_id: object,
    *,
    ontology: object | None = None,
    filter_policy: object | None = None,
    resolution_config: object | None = None,
    assembly_config: object | None = None,
    embedding_model_signature: str | None = None,
    document_artifacts_override: tuple[object, ...] | None = None,
    evaluation_request_id: uuid.UUID | None = None,
) -> _CollectionContext:
    from apps.collections.models import Collection
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    from apps.knowledge_graph.extraction.pipeline import (
        StaleSourceError,
        _ordered_chunks,
        _validate_source,
    )
    from apps.knowledge_graph.graph.assembly import (
        AssemblyConfig,
        assembly_config_checksum,
    )
    from apps.knowledge_graph.graph.filtering import (
        FilterPolicy,
        filter_policy_checksum,
    )
    from apps.knowledge_graph.models import DocumentEntity, GraphArtifact
    from apps.knowledge_graph.models.inputs import (
        collection_input_source_signature,
        collection_manifest_source_hash,
        document_membership_signature,
    )
    from apps.knowledge_graph.resolution import COLLECTION_RESOLVER_VERSION
    from apps.knowledge_graph.resolution.collection import (
        CollectionResolutionConfig,
        resolution_config_checksum,
    )
    from aquillm.utils import strict_index_embedding_signature

    if type(collection_id) is not int or not 0 < collection_id < 2**63:
        raise ValueError("collection id must be a positive database integer")
    collection = Collection.objects.filter(pk=collection_id).first()
    if collection is None:
        raise StaleBuildError("collection no longer exists")
    ontology = _active_ontology(collection_id) if ontology is None else ontology
    ontology_activation_signature = _ontology_activation_signature(ontology)
    filter_policy = FilterPolicy() if filter_policy is None else filter_policy
    resolution_config = (
        CollectionResolutionConfig() if resolution_config is None else resolution_config
    )
    assembly_config = AssemblyConfig() if assembly_config is None else assembly_config
    document_cap = min(
        resolution_config.max_document_inputs,
        assembly_config.max_document_inputs,
    )
    embedding_model_signature = (
        strict_index_embedding_signature()
        if embedding_model_signature is None
        else embedding_model_signature
    )
    document_models = tuple(
        sorted(DESCENDED_FROM_DOCUMENT, key=lambda value: value._meta.label)
    )
    document_count = sum(
        model.objects.filter(
            collection_id=collection_id,
            ingestion_complete=True,
        ).count()
        for model in document_models
    )
    _validate_collection_context_caps(
        document_count=document_count,
        entity_count=0,
        resolution_config=resolution_config,
        assembly_config=assembly_config,
    )
    document_id_values: list[object] = []
    for model in document_models:
        remaining = document_cap - len(document_id_values)
        rows = _bounded_context_rows(
            model.objects.filter(
                collection_id=collection_id,
                ingestion_complete=True,
            )
            .order_by("id")
            .values_list("id", flat=True),
            max(remaining, 1),
            "collection document",
        )
        document_id_values.extend(rows)
        if len(document_id_values) > document_cap:
            raise CorruptBuildError("collection document cap exceeded")
    document_ids = tuple(document_id_values)
    if len(document_ids) != len(set(document_ids)):
        raise CorruptBuildError("collection contains duplicate concrete document UUIDs")
    if document_artifacts_override is None:
        if evaluation_request_id is not None:
            raise CorruptBuildError(
                "evaluation collection context requires exact document artifacts"
            )
        artifacts = _bounded_context_rows(
            GraphArtifact.objects.filter(
                scope_type=GraphArtifact.ScopeType.DOCUMENT,
                scope_id__in=tuple(map(str, document_ids)),
                status=GraphArtifact.Status.ACTIVE,
                evaluation_only=False,
            ).order_by("pk"),
            document_cap,
            "collection document artifact",
        )
    else:
        if type(document_artifacts_override) is not tuple:
            raise CorruptBuildError(
                "collection document artifact override must be an exact tuple"
            )
        if len(document_artifacts_override) > document_cap:
            raise CorruptBuildError("collection document artifact cap exceeded")
        if (
            evaluation_request_id is not None
            and type(evaluation_request_id) is not uuid.UUID
        ):
            raise CorruptBuildError("evaluation request id must be an exact UUID")
        artifacts = tuple(sorted(document_artifacts_override, key=lambda row: row.pk))
        if any(type(row) is not GraphArtifact or row.pk is None for row in artifacts):
            raise CorruptBuildError(
                "collection document artifact override contains a forged row"
            )
        if evaluation_request_id is None:
            eligible = all(
                row.scope_type == GraphArtifact.ScopeType.DOCUMENT
                and row.status == GraphArtifact.Status.ACTIVE
                and row.evaluation_only is False
                for row in artifacts
            )
        else:
            eligible = all(
                row.scope_type == GraphArtifact.ScopeType.DOCUMENT
                and row.status == GraphArtifact.Status.SUPERSEDED
                and row.evaluation_only is True
                and row.rebuild_request_id == evaluation_request_id
                for row in artifacts
            )
        if not eligible:
            raise CorruptBuildError(
                "collection document artifact override is not eligible"
            )
    artifact_document_ids = tuple(artifact.scope_id for artifact in artifacts)
    if len(artifact_document_ids) != len(set(artifact_document_ids)):
        raise CorruptBuildError("collection has duplicate active document artifacts")
    if set(artifact_document_ids) != {str(value) for value in document_ids}:
        raise StaleBuildError(
            "collection awaits a graph artifact for every eligible document"
        )
    if any(
        artifact.ontology_version != ontology.version
        or artifact.ontology_checksum != ontology.checksum
        for artifact in artifacts
    ):
        raise StaleBuildError("collection awaits fresh document graph artifacts")
    entity_count = DocumentEntity.objects.filter(
        artifact_id__in=tuple(artifact.pk for artifact in artifacts),
        status=DocumentEntity.Status.ACTIVE,
    ).count()
    _validate_collection_context_caps(
        document_count=document_count,
        entity_count=entity_count,
        resolution_config=resolution_config,
        assembly_config=assembly_config,
    )
    artifact_document_uuid_set = {uuid.UUID(value) for value in artifact_document_ids}
    document_values: list[object] = []
    for model in document_models:
        remaining = document_cap - len(document_values)
        rows = _bounded_context_rows(
            model.objects.filter(
                id__in=artifact_document_uuid_set,
                collection_id=collection_id,
                ingestion_complete=True,
            ).order_by("pk"),
            max(remaining, 1),
            "collection concrete document",
        )
        document_values.extend(rows)
        if len(document_values) > document_cap:
            raise CorruptBuildError("collection concrete document cap exceeded")
    documents = tuple(document_values)
    documents_by_id = {str(document.id): document for document in documents}
    if set(documents_by_id) != set(artifact_document_ids):
        raise StaleBuildError("active document artifact escaped collection membership")
    for artifact in artifacts:
        document = documents_by_id[artifact.scope_id]
        metadata = artifact.metadata if type(artifact.metadata) is dict else {}
        current_chunks = _ordered_chunks(document.id)
        current_chunk_signature = ordered_chunk_signature(
            current_chunks,
            concrete_model_label=document._meta.label_lower,
        )
        try:
            _validate_source(document, artifact.source_hash)
        except StaleSourceError as exc:
            raise StaleBuildError(
                "collection awaits fresh document graph artifacts"
            ) from exc
        if not (
            artifact.orchestration_version
            != GraphArtifact.OrchestrationVersion.SCOPED_V1
            or (
                metadata.get("ordered_chunk_signature") == current_chunk_signature
                and metadata.get("ontology_activation_signature")
                == ontology_activation_signature
            )
        ):
            raise StaleBuildError("collection awaits fresh document graph artifacts")
    contributing = artifacts
    source_signatures = []
    for artifact in contributing:
        document = documents_by_id[artifact.scope_id]
        membership = document_membership_signature(document)
        source_signatures.append(
            collection_input_source_signature(
                collection_id=collection_id,
                document_id=document.id,
                document_artifact=artifact,
                membership_signature=membership,
            )
        )
    aggregate = collection_manifest_source_hash(source_signatures)
    extractor_version = _collection_extractor_version(contributing)
    identity = CollectionBuildIdentity(
        collection_id=collection_id,
        aggregate_source_signature=aggregate,
        extractor_version=extractor_version,
        ontology_version=ontology.version,
        ontology_checksum=ontology.checksum,
        resolver_version=COLLECTION_RESOLVER_VERSION,
        resolver_checksum=resolution_config_checksum(resolution_config),
        filter_version=filter_policy.version,
        filter_checksum=filter_policy_checksum(filter_policy),
        assembly_version=assembly_config.version,
        assembly_checksum=assembly_config_checksum(assembly_config),
        embedding_model_signature=embedding_model_signature,
        ontology_activation_signature=ontology_activation_signature,
    )
    return _CollectionContext(
        identity=identity,
        collection=collection,
        document_artifacts=contributing,
        ontology=ontology,
        filter_policy=filter_policy,
        resolution_config=resolution_config,
        assembly_config=assembly_config,
    )


def _collection_context_for_request(
    collection_id: int,
    request: object | None,
) -> _CollectionContext:
    if request is None or not request.evaluation_only:
        return _collection_context(collection_id)
    expected = tuple(request.requested_documents)
    artifacts = _completed_request_document_artifacts(
        request,
        expected,
        for_update=False,
    )
    if len(artifacts) != len(expected):
        raise StaleBuildError(
            "evaluation collection awaits its correlated document occurrences"
        )
    return _collection_context(
        collection_id,
        document_artifacts_override=artifacts,
        evaluation_request_id=request.pk,
    )


def _lock_collection_build_rows(
    collection_id: int,
    *,
    build_key: str,
    candidate_artifact_id: int | None = None,
):
    from apps.knowledge_graph.graph.assembly import lock_collection_graph_scope
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    collection = lock_collection_graph_scope(collection_id)
    scope_query = GraphArtifact.objects.filter(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=str(collection_id),
    )
    artifact_ids = _bounded_scope_artifact_ids(
        scope_query,
        build_key=build_key,
        candidate_artifact_id=candidate_artifact_id,
    )
    artifacts = tuple(
        GraphArtifact.objects.select_for_update()
        .filter(pk__in=artifact_ids)
        .order_by("pk")
    )
    run_ids = tuple(
        GraphBuildRun.objects.filter(
            artifact_id__in=artifact_ids,
            orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        )
        .order_by("pk")
        .values_list("pk", flat=True)[: len(artifact_ids) + 1]
    )
    runs = tuple(
        GraphBuildRun.objects.select_for_update().filter(pk__in=run_ids).order_by("pk")
    )
    return collection, artifacts, runs


def _revalidate_active_collection_build(
    context: _CollectionContext,
    collection: object,
    artifact: object,
    run: object,
    build_key: str,
) -> None:
    """Linearize an active fast path against its locked live manifest."""

    from apps.knowledge_graph.graph.assembly import (
        CollectionGraphAssemblyError,
        CollectionGraphSourceStaleError,
        validate_locked_active_collection_snapshot,
    )

    identity = context.identity
    if (
        derive_collection_build_key(context.identity) != build_key
        or artifact.build_key != build_key
        or run.build_key != build_key
        or artifact.source_hash != identity.aggregate_source_signature
        or artifact.ontology_version != identity.ontology_version
        or artifact.ontology_checksum != identity.ontology_checksum
        or artifact.extractor_version != identity.extractor_version
        or artifact.resolver_version != identity.resolver_version
        or artifact.filter_policy_version != identity.filter_version
        or artifact.filter_policy_checksum != identity.filter_checksum
        or artifact.resolution_config_checksum != identity.resolver_checksum
        or artifact.assembly_version != identity.assembly_version
        or artifact.assembly_config_checksum != identity.assembly_checksum
        or artifact.embedding_model_signature != identity.embedding_model_signature
    ):
        raise CorruptBuildError(
            "active collection occurrence differs from the requested build identity"
        )
    try:
        validate_locked_active_collection_snapshot(
            collection=collection,
            artifact=artifact,
            run=run,
            aggregate_source_signature=identity.aggregate_source_signature,
            ontology=context.ontology,
            config=context.assembly_config,
        )
    except CollectionGraphSourceStaleError:
        raise
    except CollectionGraphAssemblyError as exc:
        raise CorruptBuildError("active collection snapshot is corrupt") from exc


def _bootstrap_collection_build(
    context: _CollectionContext,
    build_key: str,
    request_id: uuid.UUID | None = None,
    eval_only: bool = False,
) -> tuple[object, object, str | None, int | None, bool]:
    from apps.knowledge_graph.graph.assembly import CollectionGraphSourceStaleError
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )
    from apps.knowledge_graph.resolution.collection import build_collection_snapshot

    owner = uuid.uuid4().hex
    stale_active = False
    with transaction.atomic():
        request = None
        if request_id is not None:
            try:
                _parent, request = _lock_rebuild_request_prefix(request_id)
            except LookupError as exc:
                raise StaleBuildError("rebuild request no longer exists") from exc
        collection, artifacts, runs = _lock_collection_build_rows(
            context.identity.collection_id,
            build_key=build_key,
        )
        scope_runs = _lock_latest_scope_run(
            GraphBuildRun.BuildKind.COLLECTION,
            context.identity.collection_id,
        )
        if request_id is not None:
            assert request is not None
            if (
                request.scope_type != GraphRebuildRequest.ScopeType.COLLECTION
                or request.scope_id != str(context.identity.collection_id)
                or request.expected_aggregate_signature
                != context.identity.aggregate_source_signature
            ):
                raise StaleBuildError("collection is outside the rebuild snapshot")
            if request.evaluation_only is not eval_only:
                raise CorruptBuildError("rebuild evaluation marker changed")
            correlated = tuple(
                GraphArtifact.objects.select_for_update()
                .filter(
                    rebuild_request_id=request.pk,
                    scope_type=GraphArtifact.ScopeType.COLLECTION,
                    scope_id=str(context.identity.collection_id),
                )
                .order_by("pk")[:2]
            )
            if len(correlated) > 1:
                raise CorruptBuildError("request owns duplicate collection occurrences")
            for row in correlated:
                if all(existing.pk != row.pk for existing in artifacts):
                    artifacts = (*artifacts, row)
            correlated_run_rows = tuple(
                GraphBuildRun.objects.select_for_update()
                .filter(
                    rebuild_request_id=request.pk,
                    build_kind=GraphBuildRun.BuildKind.COLLECTION,
                    scope_id=str(context.identity.collection_id),
                )
                .order_by("pk")[:2]
            )
            if len(correlated_run_rows) > 1:
                raise CorruptBuildError("request owns duplicate collection build runs")
            for row in correlated_run_rows:
                if all(existing.pk != row.pk for existing in runs):
                    runs = (*runs, row)
            if request.status in {
                GraphRebuildRequest.Status.SUCCEEDED,
                GraphRebuildRequest.Status.FAILED,
                GraphRebuildRequest.Status.PARTIAL,
            }:
                if correlated and correlated_run_rows:
                    return correlated[0], correlated_run_rows[0], None, None, True
                raise StaleBuildError("rebuild request is already terminal")
            if correlated and correlated[0].status == GraphArtifact.Status.SUPERSEDED:
                completed_run = (
                    correlated_run_rows[0] if len(correlated_run_rows) == 1 else None
                )
                if (
                    request.evaluation_only
                    and _evaluation_occurrence_completed(
                        correlated[0],
                        completed_run,
                        build_kind=GraphBuildRun.BuildKind.COLLECTION,
                    )
                ) or (
                    not request.evaluation_only
                    and _production_occurrence_completed(
                        correlated[0],
                        completed_run,
                        build_kind=GraphBuildRun.BuildKind.COLLECTION,
                        allow_historical=True,
                    )
                ):
                    return (
                        correlated[0],
                        correlated_run_rows[0],
                        None,
                        None,
                        True,
                    )
                raise StaleBuildError("request collection occurrence was superseded")
            action = (
                _occurrence_action(correlated, correlated_run_rows, build_key)
                if correlated
                else OccurrenceAction.CREATE
            )
            action_artifacts = correlated
            action_runs = correlated_run_rows
        else:
            action_artifacts = tuple(
                row for row in artifacts if row.evaluation_only is False
            )
            action_artifact_ids = {row.pk for row in action_artifacts}
            action_runs = tuple(
                row
                for row in runs
                if row.evaluation_only is False
                and row.artifact_id in action_artifact_ids
            )
            action = _occurrence_action(action_artifacts, action_runs, build_key)
        run_by_artifact = {row.artifact_id: row for row in action_runs}
        artifact = None
        run = None
        if action is OccurrenceAction.RETURN_ACTIVE:
            artifact = next(
                row
                for row in action_artifacts
                if row.build_key == build_key
                and row.status == GraphArtifact.Status.ACTIVE
                and row.orchestration_version
                == GraphArtifact.OrchestrationVersion.SCOPED_V1
            )
            run = run_by_artifact[artifact.pk]
            if run.lease_owner or run.lease_expires_at is not None:
                raise CorruptBuildError("active collection owns a build lease")
            try:
                _revalidate_active_collection_build(
                    context,
                    collection,
                    artifact,
                    run,
                    build_key,
                )
            except CollectionGraphSourceStaleError:
                if request is None:
                    transaction.on_commit(
                        lambda: enqueue_current_collection_refresh(
                            context.identity.collection_id
                        ),
                        robust=True,
                    )
                else:
                    transaction.on_commit(
                        lambda: record_rebuild_failure(
                            request.pk,
                            error_code="collection_identity_changed",
                            resnapshot=True,
                        ),
                        robust=True,
                    )
                stale_active = True
            else:
                return artifact, run, None, None, True
        if stale_active:
            pass
        elif action in {OccurrenceAction.RESUME, OccurrenceAction.RETRY}:
            artifact = max(
                (
                    row
                    for row in action_artifacts
                    if row.build_key == build_key
                    and row.orchestration_version
                    == GraphArtifact.OrchestrationVersion.SCOPED_V1
                ),
                key=lambda row: (row.build_generation, row.pk),
            )
            run = run_by_artifact[artifact.pk]
            _validate_retryable_run(run)
            if _run_has_live_lease(run):
                raise BuildInProgressError(
                    "exact collection graph build already has a live lease"
                )
            _restart_locked_run(run)
            if artifact.status in {
                GraphArtifact.Status.FAILED,
                GraphArtifact.Status.STALE,
            }:
                artifact.status = GraphArtifact.Status.BUILDING
                artifact.save(update_fields=["status"])
            elif artifact.status != GraphArtifact.Status.BUILDING:
                raise CorruptBuildError("collection retry artifact is not reusable")
        else:
            build_generation = _next_build_generation(artifacts, scope_runs)
            artifact, _manifest = build_collection_snapshot(
                collection=context.collection,
                document_artifacts=context.document_artifacts,
                ontology=context.ontology,
                extractor_version=context.identity.extractor_version,
                resolver_version=context.identity.resolver_version,
                filter_policy=context.filter_policy,
                resolution_config=context.resolution_config,
                assembly_config=context.assembly_config,
                embedding_model_signature=(context.identity.embedding_model_signature),
                build_key=build_key,
                build_generation=build_generation,
                orchestration_version=(GraphArtifact.OrchestrationVersion.SCOPED_V1),
                rebuild_request=request,
                evaluation_only=eval_only,
            )
            artifact.metadata = {
                **(artifact.metadata if type(artifact.metadata) is dict else {}),
                "orchestration_version": 1,
                "build_key": build_key,
                "ontology_activation_signature": (
                    context.identity.ontology_activation_signature
                ),
            }
            artifact.save(update_fields=["metadata"])
            if artifact.source_hash != context.identity.aggregate_source_signature:
                raise StaleBuildError("collection manifest changed during snapshot")
            run = GraphBuildRun.objects.create(
                artifact=artifact,
                rebuild_request=request,
                evaluation_only=eval_only,
                build_generation=artifact.build_generation,
                orchestration_version=artifact.orchestration_version,
                stage=GraphBuildRun.Stage.QUEUED,
                status=GraphBuildRun.Status.PENDING,
                attempt=1,
                metadata={
                    "orchestration_version": 1,
                    "attempt_history": [],
                },
                stage_marker={
                    "orchestration_version": 1,
                    "build_key": build_key,
                    "aggregate_source_signature": (
                        context.identity.aggregate_source_signature
                    ),
                    "stage_sequence": [GraphBuildRun.Stage.QUEUED],
                    "last_stage": GraphBuildRun.Stage.QUEUED,
                },
            )
        if not stale_active:
            lease_owner, lease_generation = _claim_locked_run(run, owner)
            return artifact, run, lease_owner, lease_generation, False
    raise StaleBuildError("active collection graph no longer matches live contributors")


def _terminal_collection_build(
    context: _CollectionContext,
    artifact_id: int,
    run_id: int,
    *,
    lease_owner: str,
    lease_generation: int,
    stale: bool,
    error_code: str,
    reschedule: bool = False,
) -> None:
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    with transaction.atomic():
        artifact, run = _lock_terminal_collection_rows(
            context.identity.collection_id,
            artifact_id,
            run_id,
        )
        if artifact is None or run is None:
            return
        validate_build_lease(run, lease_owner, lease_generation)
        target = GraphBuildRun.Stage.STALE if stale else GraphBuildRun.Stage.FAILED
        artifact.status = (
            GraphArtifact.Status.STALE if stale else GraphArtifact.Status.FAILED
        )
        artifact.completed_at = timezone.now()
        artifact.save(update_fields=["status", "completed_at"])
        _apply_locked_terminal(run, target, error_code=error_code)
        if stale and reschedule:
            if run.rebuild_request_id is None:
                transaction.on_commit(
                    lambda: enqueue_current_collection_refresh(
                        context.identity.collection_id
                    ),
                    robust=True,
                )
            else:
                transaction.on_commit(
                    lambda: record_rebuild_failure(
                        run.rebuild_request_id,
                        error_code=error_code,
                        resnapshot=True,
                    ),
                    robust=True,
                )


def refresh_collection_graph(
    collection_id,
    aggregate_source_signature,
    collection_build_key,
    request_id=None,
    eval_only=False,
):
    """Build and atomically activate one exact collection graph snapshot."""

    request = _validated_rebuild_request_before_content(
        request_id,
        eval_only,
        build_kind="collection",
        scope_id=collection_id,
        source_hash=aggregate_source_signature,
    )
    resolved_request_id = None if request is None else request.pk
    from apps.knowledge_graph.graph.assembly import (
        CollectionGraphSourceStaleError,
        activate_collection_graph,
        assemble_collection_graph,
        validate_collection_graph_artifact,
    )
    from apps.knowledge_graph.graph.filtering import filter_collection_resolution
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.resolution.collection import (
        default_collection_embedding_session,
        load_collection_filter_inputs,
        load_collection_resolution_inputs,
        persist_collection_resolution,
        resolve_collection_entities,
    )

    started = perf_counter()
    if request is not None and (
        request.scope_type != request.ScopeType.COLLECTION
        or request.scope_id != str(collection_id)
    ):
        raise StaleBuildError("collection is outside the rebuild request scope")
    context = _collection_context_for_request(collection_id, request)
    expected_aggregate = _hash(aggregate_source_signature, "aggregate source signature")
    requested_key = _hash(collection_build_key, "collection build key")
    if context.identity.aggregate_source_signature != expected_aggregate:
        if resolved_request_id is None:
            enqueue_collection_refresh(
                context.identity.collection_id,
                context.identity.aggregate_source_signature,
                derive_collection_build_key(context.identity),
            )
        else:
            record_rebuild_failure(
                resolved_request_id,
                error_code="collection_identity_changed",
                resnapshot=True,
            )
        raise StaleBuildError("collection aggregate source signature is stale")
    if derive_collection_build_key(context.identity) != requested_key:
        if resolved_request_id is None:
            enqueue_collection_refresh(
                context.identity.collection_id,
                context.identity.aggregate_source_signature,
                derive_collection_build_key(context.identity),
            )
        else:
            record_rebuild_failure(
                resolved_request_id,
                error_code="collection_identity_changed",
                resnapshot=True,
            )
        raise StaleBuildError("collection build key does not match live manifest")
    if resolved_request_id is None and not eval_only:
        artifact, run, lease_owner, lease_generation, completed = (
            _bootstrap_collection_build(context, requested_key)
        )
    else:
        artifact, run, lease_owner, lease_generation, completed = (
            _bootstrap_collection_build(
                context,
                requested_key,
                resolved_request_id,
                eval_only,
            )
        )
    if completed:
        complete_collection_rebuild(resolved_request_id, artifact)
        return artifact
    assert lease_owner is not None and lease_generation is not None
    logger.info(
        "obs.kg.build_started",
        build_kind="collection",
        scope_id=str(collection_id),
        build_key=requested_key,
        artifact_id=artifact.pk,
        build_run_id=run.pk,
        attempt=run.attempt,
        ontology_version=context.identity.ontology_version,
        resolver_version=context.identity.resolver_version,
        filter_version=context.identity.filter_version,
        assembly_version=context.identity.assembly_version,
    )
    try:
        if run.stage == GraphBuildRun.Stage.QUEUED:
            run = _transition_run(
                run.pk,
                GraphBuildRun.Stage.SNAPSHOTTING,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
                marker={
                    "document_artifact_count": len(context.document_artifacts),
                },
            )
        if run.stage == GraphBuildRun.Stage.SNAPSHOTTING:
            run = _transition_run(
                run.pk,
                GraphBuildRun.Stage.RESOLVING,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
            )
        if run.stage == GraphBuildRun.Stage.RESOLVING:
            resolution_state = _collection_resolution_commit_state(
                context,
                artifact,
                run,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
            )
            if resolution_state is CommitMarkerState.CORRUPT:
                raise CorruptBuildError("collection resolution commit is corrupt")
            if resolution_state is CommitMarkerState.ABSENT:
                with LeaseHeartbeat(run.pk, lease_owner, lease_generation):
                    snapshot, entities, relations = load_collection_resolution_inputs(
                        artifact.pk,
                        run.pk,
                        config=context.resolution_config,
                        lease_owner=lease_owner,
                        lease_generation=lease_generation,
                    )
                    resolution = resolve_collection_entities(
                        snapshot,
                        entities,
                        context.ontology,
                        relations=relations,
                        config=context.resolution_config,
                        embedding_session=default_collection_embedding_session(
                            context.identity.embedding_model_signature
                        ),
                    )
                    filter_inputs = load_collection_filter_inputs(
                        artifact.pk,
                        run.pk,
                        resolution,
                        lease_owner=lease_owner,
                        lease_generation=lease_generation,
                    )
                    filter_result = filter_collection_resolution(
                        resolution,
                        filter_inputs,
                        context.ontology,
                        context.filter_policy,
                    )
                    persist_collection_resolution(
                        artifact.pk,
                        run.pk,
                        resolution,
                        filter_result,
                        filter_policy=context.filter_policy,
                        ontology=context.ontology,
                        lease_owner=lease_owner,
                        lease_generation=lease_generation,
                    )
            run = _transition_run(
                run.pk,
                GraphBuildRun.Stage.ASSEMBLING,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
            )
        if run.stage == GraphBuildRun.Stage.ASSEMBLING:
            assembly_state = _collection_assembly_commit_state(
                context,
                artifact,
                run,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
            )
            if assembly_state is CommitMarkerState.CORRUPT:
                raise CorruptBuildError("collection assembly commit is corrupt")
            if assembly_state is CommitMarkerState.ABSENT:
                with LeaseHeartbeat(run.pk, lease_owner, lease_generation):
                    assemble_collection_graph(
                        collection_id,
                        run.pk,
                        expected_aggregate,
                        ontology=context.ontology,
                        config=context.assembly_config,
                        lease_owner=lease_owner,
                        lease_generation=lease_generation,
                    )
            run = _transition_run(
                run.pk,
                GraphBuildRun.Stage.VALIDATING,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
            )
        if run.stage == GraphBuildRun.Stage.VALIDATING:
            validate_collection_graph_artifact(
                collection_id,
                run.pk,
                expected_aggregate,
                ontology=context.ontology,
                config=context.assembly_config,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
            )
        current = _collection_context_for_request(collection_id, request)
        current_key = derive_collection_build_key(current.identity)
        if current.identity != context.identity or current_key != requested_key:
            try:
                _terminal_collection_build(
                    context,
                    artifact.pk,
                    run.pk,
                    lease_owner=lease_owner,
                    lease_generation=lease_generation,
                    stale=True,
                    error_code="collection_identity_changed",
                    reschedule=True,
                )
            except Exception:
                logger.error(
                    "obs.kg.build_terminal_failed",
                    build_kind="collection",
                    scope_id=str(collection_id),
                    build_key=requested_key,
                    artifact_id=artifact.pk,
                    build_run_id=run.pk,
                    attempt=run.attempt,
                    error_code="terminal_bookkeeping_failed",
                )
            raise StaleBuildError("collection manifest changed before activation")
        activate_collection_graph(
            collection_id,
            run.pk,
            expected_aggregate,
            ontology=context.ontology,
            config=context.assembly_config,
            lease_owner=lease_owner,
            lease_generation=lease_generation,
        )
        artifact = GraphArtifact.objects.get(pk=artifact.pk)
        run = GraphBuildRun.objects.get(pk=run.pk)
        complete_collection_rebuild(resolved_request_id, artifact)
        logger.info(
            "obs.kg.build_completed",
            build_kind="collection",
            scope_id=str(collection_id),
            build_key=requested_key,
            artifact_id=artifact.pk,
            build_run_id=run.pk,
            attempt=run.attempt,
            document_artifact_count=len(context.document_artifacts),
            total_seconds=perf_counter() - started,
        )
        return artifact
    except Exception as exc:
        if isinstance(exc, StaleBuildError):
            # The explicit drift branch already committed stale state and its
            # exact replacement callback. Other stale errors are fenced here.
            try:
                run.refresh_from_db()
            except Exception:
                pass
            if run.stage == GraphBuildRun.Stage.STALE:
                raise
        replacement = None
        replacement_key = None
        stale = isinstance(exc, CollectionGraphSourceStaleError)
        try:
            replacement = _collection_context_for_request(collection_id, request)
            replacement_key = derive_collection_build_key(replacement.identity)
            stale = stale or (
                replacement.identity != context.identity
                or replacement_key != requested_key
            )
        except Exception:
            replacement = None
        error_code = (
            "collection_identity_changed"
            if stale
            else (
                "corrupt_build_state"
                if isinstance(exc, CorruptBuildError)
                else "collection_build_failed"
            )
        )
        try:
            _terminal_collection_build(
                context,
                artifact.pk,
                run.pk,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
                stale=stale,
                error_code=error_code,
                reschedule=(
                    stale
                    and replacement_key is not None
                    and replacement_key != requested_key
                ),
            )
        except Exception:
            logger.error(
                "obs.kg.build_terminal_failed",
                build_kind="collection",
                scope_id=str(collection_id),
                build_key=requested_key,
                artifact_id=artifact.pk,
                build_run_id=run.pk,
                attempt=run.attempt,
                error_code="terminal_bookkeeping_failed",
            )
        if stale or isinstance(exc, CorruptBuildError):
            record_rebuild_failure(
                resolved_request_id,
                error_code=error_code,
                resnapshot=stale,
            )
        logger.error(
            "obs.kg.build_failed",
            build_kind="collection",
            scope_id=str(collection_id),
            build_key=requested_key,
            artifact_id=artifact.pk,
            build_run_id=run.pk,
            attempt=run.attempt,
            stage=run.stage,
            error_code=error_code,
            total_seconds=perf_counter() - started,
        )
        raise


__all__ = [
    "BUILD_LEASE_RETRY_SECONDS",
    "BuildInProgressError",
    "BuildLeaseLostError",
    "CollectionBuildIdentity",
    "CorruptBuildError",
    "DocumentBuildIdentity",
    "StaleBuildError",
    "advance_rebuild_request",
    "assert_evaluation_bypass",
    "build_document_graph",
    "complete_collection_rebuild",
    "create_rebuild_request",
    "derive_current_document_build_key",
    "derive_collection_build_key",
    "derive_document_build_key",
    "enqueue_collection_refresh",
    "enqueue_current_collection_refresh",
    "enqueue_document_build",
    "preview_rebuild",
    "record_rebuild_failure",
    "refresh_collection_graph",
    "validate_rebuild_task_request_metadata",
    "validate_orchestration_stage",
    "validate_build_lease",
    "validate_stage_transition",
]
