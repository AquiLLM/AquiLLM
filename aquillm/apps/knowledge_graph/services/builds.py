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
from django.db.models import DateTimeField, ExpressionWrapper, F, Q, Value
from django.db.models.functions import Now
from django.utils import timezone

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LEASE_DURATION = timedelta(minutes=30)
BUILD_LEASE_RETRY_SECONDS = int(_LEASE_DURATION.total_seconds()) + 30
_DOCUMENT_LOCK_NAMESPACE = 0x4B47
_EXTRACTOR_PACKAGE_IDENTITY = "gliner2==1.3.2"
_GRAPH_TASK_PUBLISH_RETRY_POLICY = {
    "max_retries": 3,
    "interval_start": 0,
    "interval_step": 0.5,
    "interval_max": 5,
}

logger = structlog.stdlib.get_logger(__name__)


class BuildLeaseLostError(RuntimeError):
    """The caller no longer owns the durable attempt generation."""


class BuildInProgressError(RuntimeError):
    """Another live worker currently owns this exact build identity."""


class StaleBuildError(RuntimeError):
    """The immutable requested source no longer matches live source state."""


class CorruptBuildError(RuntimeError):
    """Persisted rows cannot be tied to a complete commit marker."""


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


def _next_build_generation(artifacts) -> int:
    generations = tuple(getattr(row, "build_generation", None) for row in artifacts)
    if any(type(value) is not int or value < 1 for value in generations):
        raise CorruptBuildError("build occurrence generation is invalid")
    if len(generations) != len(set(generations)):
        raise CorruptBuildError("scope owns duplicate build generations")
    return max(generations, default=0) + 1


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


def _active_ontology():
    from apps.knowledge_graph.models import OntologyVersion
    from apps.knowledge_graph.services.ontology import load_ontology_yaml

    records = tuple(
        OntologyVersion.objects.filter(
            kind=OntologyVersion.Kind.GRAPH,
            status=OntologyVersion.Status.ACTIVE,
        ).order_by("pk")[:2]
    )
    if len(records) != 1:
        raise StaleBuildError("graph build requires exactly one active ontology")
    record = records[0]
    metadata = record.metadata if type(record.metadata) is dict else {}
    raw_yaml = metadata.get("yaml")
    if type(raw_yaml) is not str:
        raise StaleBuildError("active ontology has no immutable YAML snapshot")
    definition = load_ontology_yaml(raw_yaml)
    if definition.version != record.version or definition.checksum != record.checksum:
        raise StaleBuildError("active ontology identity does not match its YAML")
    return definition


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
    try:
        _validate_source(document, source_hash)
    except Exception as exc:
        raise StaleBuildError("document source hash changed") from exc
    chunks = _ordered_chunks(document_id, for_update=for_update)
    chunk_signature = ordered_chunk_signature(
        chunks,
        concrete_model_label=document._meta.label_lower,
    )
    ontology = _active_ontology() if ontology is None else ontology
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
    collection_id = getattr(document, "collection_id", None)
    if type(collection_id) is not int or collection_id <= 0:
        raise StaleBuildError("document has no concrete collection membership")
    return _DocumentContext(
        identity=identity,
        collection_id=collection_id,
        ontology=ontology,
        settings=settings,
    )


def _lock_document_scope(document_id: uuid.UUID) -> None:
    if connection.vendor != "postgresql":
        return
    lock_key = (
        int.from_bytes(sha256(document_id.bytes).digest()[:4], "big", signed=False)
        & 0x7FFFFFFF
    )
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_xact_lock(%s, %s)",
            [_DOCUMENT_LOCK_NAMESPACE, lock_key],
        )


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
            GraphBuildRun.Stage.VALIDATING: frozenset({GraphBuildRun.Stage.ACTIVE})
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
            GraphBuildRun.Stage.VALIDATING: frozenset({GraphBuildRun.Stage.ACTIVE})
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


def enqueue_document_build(
    document_id: uuid.UUID,
    expected_source_hash: str,
) -> None:
    """Publish one provider-neutral, JSON-safe document build request."""

    from lib.knowledge_graph.config import get_build_enabled

    if not get_build_enabled():
        return
    if type(document_id) is not uuid.UUID:
        raise ValueError("document id must be an exact UUID")
    source_hash = _hash(expected_source_hash, "expected source hash")
    build_key = derive_current_document_build_key(document_id, source_hash)
    from apps.knowledge_graph.tasks import build_document_graph_task

    _publish_graph_task(
        build_document_graph_task,
        kwargs={
            "document_id": str(document_id),
            "expected_source_hash": source_hash,
            "document_build_key": build_key,
        },
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
) -> None:
    """Publish one exact, JSON-safe collection refresh request."""

    from lib.knowledge_graph.config import get_build_enabled

    if not get_build_enabled():
        return
    if type(collection_id) is not int or not 0 < collection_id < 2**63:
        raise ValueError("collection id must be a positive database integer")
    aggregate_signature = _hash(
        aggregate_source_signature,
        "aggregate source signature",
    )
    build_key = _hash(collection_build_key, "collection build key")
    from apps.knowledge_graph.tasks import refresh_collection_graph_task

    _publish_graph_task(
        refresh_collection_graph_task,
        kwargs={
            "collection_id": collection_id,
            "aggregate_source_signature": aggregate_signature,
            "collection_build_key": build_key,
        },
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


def _enqueue_current_collection_refresh(collection_id: int) -> None:
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


def _register_document_refresh_callbacks(
    context: _DocumentContext,
    run: object,
) -> None:
    metadata = run.metadata if type(run.metadata) is dict else {}
    initial_collection_id = metadata.get("initial_collection_id")
    affected = {context.collection_id}
    if type(initial_collection_id) is int and initial_collection_id > 0:
        affected.add(initial_collection_id)
    for collection_id in sorted(affected):
        transaction.on_commit(
            lambda collection_id=collection_id: _enqueue_current_collection_refresh(
                collection_id
            ),
            robust=True,
        )


def _bootstrap_document_build(
    context: _DocumentContext,
    build_key: str,
) -> tuple[object, object, str | None, int | None, bool]:
    from apps.knowledge_graph.extraction.pipeline import (
        _validate_source,
        resolve_ontology_definition,
    )
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from lib.knowledge_graph.config import load_extraction_settings

    owner = uuid.uuid4().hex
    with transaction.atomic():
        artifacts, runs, document, chunks = _lock_document_build_rows(
            context.identity.document_id,
            build_key=build_key,
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

        action = _occurrence_action(artifacts, runs, build_key)
        run_by_artifact = {row.artifact_id: row for row in runs}
        artifact = None
        run = None
        if action is OccurrenceAction.RETURN_ACTIVE:
            artifact = next(
                row
                for row in artifacts
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
                    for row in artifacts
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
            build_generation = _next_build_generation(artifacts)
            artifact = GraphArtifact.objects.create(
                status=GraphArtifact.Status.BUILDING,
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
            if row.status == GraphArtifact.Status.ACTIVE and row.pk != artifact.pk
        )
        if len(active) > 1:
            raise CorruptBuildError("document scope has multiple active artifacts")
        higher_current = tuple(
            row
            for row in artifacts
            if row.build_generation > artifact.build_generation
            and row.status
            in {GraphArtifact.Status.BUILDING, GraphArtifact.Status.ACTIVE}
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


def build_document_graph(document_id, expected_source_hash, document_build_key):
    """Build and atomically activate one exact document graph."""

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
    artifact, run, lease_owner, lease_generation, completed = _bootstrap_document_build(
        context, requested_key
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
    ontology = _active_ontology() if ontology is None else ontology
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
        model.objects.filter(collection_id=collection_id).count()
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
            model.objects.filter(collection_id=collection_id)
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
    artifacts = _bounded_context_rows(
        GraphArtifact.objects.filter(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id__in=tuple(map(str, document_ids)),
            status=GraphArtifact.Status.ACTIVE,
        ).order_by("pk"),
        document_cap,
        "collection document artifact",
    )
    artifact_document_ids = tuple(artifact.scope_id for artifact in artifacts)
    if len(artifact_document_ids) != len(set(artifact_document_ids)):
        raise CorruptBuildError("collection has duplicate active document artifacts")
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
) -> tuple[object, object, str | None, int | None, bool]:
    from apps.knowledge_graph.graph.assembly import CollectionGraphSourceStaleError
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.resolution.collection import build_collection_snapshot

    owner = uuid.uuid4().hex
    stale_active = False
    with transaction.atomic():
        collection, artifacts, runs = _lock_collection_build_rows(
            context.identity.collection_id,
            build_key=build_key,
        )
        action = _occurrence_action(artifacts, runs, build_key)
        run_by_artifact = {row.artifact_id: row for row in runs}
        artifact = None
        run = None
        if action is OccurrenceAction.RETURN_ACTIVE:
            artifact = next(
                row
                for row in artifacts
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
                transaction.on_commit(
                    lambda: _enqueue_current_collection_refresh(
                        context.identity.collection_id
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
                    for row in artifacts
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
            build_generation = _next_build_generation(artifacts)
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
            transaction.on_commit(
                lambda: _enqueue_current_collection_refresh(
                    context.identity.collection_id
                ),
                robust=True,
            )


def refresh_collection_graph(
    collection_id, aggregate_source_signature, collection_build_key
):
    """Build and atomically activate one exact collection graph snapshot."""

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
    context = _collection_context(collection_id)
    expected_aggregate = _hash(aggregate_source_signature, "aggregate source signature")
    requested_key = _hash(collection_build_key, "collection build key")
    if context.identity.aggregate_source_signature != expected_aggregate:
        enqueue_collection_refresh(
            context.identity.collection_id,
            context.identity.aggregate_source_signature,
            derive_collection_build_key(context.identity),
        )
        raise StaleBuildError("collection aggregate source signature is stale")
    if derive_collection_build_key(context.identity) != requested_key:
        enqueue_collection_refresh(
            context.identity.collection_id,
            context.identity.aggregate_source_signature,
            derive_collection_build_key(context.identity),
        )
        raise StaleBuildError("collection build key does not match live manifest")
    artifact, run, lease_owner, lease_generation, completed = (
        _bootstrap_collection_build(context, requested_key)
    )
    if completed:
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
        current = _collection_context(collection_id)
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
            replacement = _collection_context(collection_id)
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
    "build_document_graph",
    "derive_current_document_build_key",
    "derive_collection_build_key",
    "derive_document_build_key",
    "enqueue_collection_refresh",
    "enqueue_document_build",
    "refresh_collection_graph",
    "validate_orchestration_stage",
    "validate_build_lease",
    "validate_stage_transition",
]
