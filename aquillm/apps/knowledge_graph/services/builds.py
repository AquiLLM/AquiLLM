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
from hashlib import sha256
from time import perf_counter

import structlog
from django.core.exceptions import ValidationError
from django.db import connection, transaction
from django.utils import timezone

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LEASE_DURATION = timedelta(minutes=30)
_DOCUMENT_LOCK_NAMESPACE = 0x4B47
_EXTRACTOR_PACKAGE_IDENTITY = "gliner2==1.3.2"

logger = structlog.stdlib.get_logger(__name__)


class BuildLeaseLostError(RuntimeError):
    """The caller no longer owns the durable attempt generation."""


class BuildInProgressError(RuntimeError):
    """Another live worker currently owns this exact build identity."""


class StaleBuildError(RuntimeError):
    """The immutable requested source no longer matches live source state."""


class CorruptBuildError(RuntimeError):
    """Persisted rows cannot be tied to a complete commit marker."""


def validate_build_lease(
    run: object,
    lease_owner: str | None,
    lease_generation: int | None,
    *,
    now=None,
) -> None:
    """Fence every mutating stage against stale or duplicate workers."""

    metadata = getattr(run, "metadata", None)
    if type(metadata) is not dict or metadata.get("orchestration_version") != 1:
        return
    if type(lease_owner) is not str or not lease_owner:
        raise BuildLeaseLostError("build lease owner is required")
    if getattr(run, "lease_owner", None) != lease_owner:
        raise BuildLeaseLostError("build lease owner no longer matches")
    if type(lease_generation) is not int or isinstance(lease_generation, bool):
        raise BuildLeaseLostError("build lease generation is required")
    if getattr(run, "lease_generation", None) != lease_generation:
        raise BuildLeaseLostError("build lease generation no longer matches")
    expires_at = getattr(run, "lease_expires_at", None)
    checked_at = timezone.now() if now is None else now
    if expires_at is None or expires_at <= checked_at:
        raise BuildLeaseLostError("build lease expired")


def _hash(value: object, label: str) -> str:
    if type(value) is not str or not _HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


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


def _lock_document_build_rows(document_id: uuid.UUID):
    """Apply the global document lock order and return its locked rows."""

    from apps.knowledge_graph.extraction.pipeline import (
        _get_concrete_document,
        _ordered_chunks,
    )
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    _lock_document_scope(document_id)
    artifacts = tuple(
        GraphArtifact.objects.select_for_update()
        .filter(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id=str(document_id),
        )
        .order_by("pk")
    )
    runs = tuple(
        GraphBuildRun.objects.select_for_update()
        .filter(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id=str(document_id),
        )
        .order_by("pk")
    )
    document = _get_concrete_document(document_id, for_update=True)
    chunks = _ordered_chunks(document_id, for_update=True)
    return artifacts, runs, document, chunks


def _safe_marker(value: object) -> dict[str, object]:
    return dict(value) if type(value) is dict else {}


def _commit_marker_present(run: object, name: str) -> bool:
    stats = run.stats if type(getattr(run, "stats", None)) is dict else {}
    return stats.get(name) is not None


def _attempt_history(run: object) -> list[dict[str, object]]:
    metadata = _safe_marker(getattr(run, "metadata", None))
    raw = metadata.get("attempt_history", [])
    history = list(raw) if type(raw) is list else []
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

    now = timezone.now()
    if (
        run.lease_owner
        and run.lease_expires_at is not None
        and run.lease_expires_at > now
    ):
        raise BuildInProgressError("exact graph build already has a live lease")
    run.lease_owner = owner
    run.lease_generation += 1
    run.lease_expires_at = now + _LEASE_DURATION
    run.save(update_fields=["lease_owner", "lease_generation", "lease_expires_at"])
    return owner, run.lease_generation


def _restart_locked_run(run: object) -> None:
    from apps.knowledge_graph.models import GraphBuildRun

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
        if status in {
            GraphBuildRun.Status.SUCCEEDED,
            GraphBuildRun.Status.FAILED,
            GraphBuildRun.Status.CANCELLED,
        }:
            run.finished_at = now
            run.lease_owner = ""
            run.lease_expires_at = None
        else:
            run.lease_expires_at = now + _LEASE_DURATION
        stage_marker = _safe_marker(run.stage_marker)
        sequence = stage_marker.get("stage_sequence", [])
        sequence = list(sequence) if type(sequence) is list else []
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


def enqueue_collection_refresh(
    collection_id: int,
    aggregate_source_signature: str | None = None,
    collection_build_key: str | None = None,
) -> None:
    """Lazy Task 12 seam; intentionally performs no task routing in Task 11."""

    logger.info(
        "obs.kg.build_stage",
        build_kind="collection",
        scope_id=str(collection_id),
        stage="refresh_requested",
        aggregate_source_signature=aggregate_source_signature,
        build_key=collection_build_key,
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
            )
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
            context.identity.document_id
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

        matching_artifacts = tuple(
            row for row in artifacts if row.build_key == build_key
        )
        if len(matching_artifacts) > 1:
            raise CorruptBuildError("document build key owns multiple artifacts")
        matching_runs = tuple(row for row in runs if row.build_key == build_key)
        if len(matching_runs) > 1:
            raise CorruptBuildError("document build key owns multiple logical runs")
        artifact = matching_artifacts[0] if matching_artifacts else None
        run = matching_runs[0] if matching_runs else None
        if (artifact is None) != (run is None):
            raise CorruptBuildError("document artifact/run ownership is incomplete")

        if run is not None:
            if run.artifact_id != artifact.pk or artifact.build_key != run.build_key:
                raise CorruptBuildError("document build ownership is inconsistent")
            if run.stage == GraphBuildRun.Stage.ACTIVE:
                if (
                    run.status != GraphBuildRun.Status.SUCCEEDED
                    or artifact.status != GraphArtifact.Status.ACTIVE
                    or run.lease_owner
                    or run.lease_expires_at is not None
                ):
                    raise CorruptBuildError(
                        "active document terminal state is inconsistent"
                    )
                _register_document_refresh_callbacks(context, run)
                return artifact, run, None, None, True
            if run.stage == GraphBuildRun.Stage.SUPERSEDED:
                if (
                    run.status != GraphBuildRun.Status.CANCELLED
                    or artifact.status != GraphArtifact.Status.SUPERSEDED
                    or run.lease_owner
                    or run.lease_expires_at is not None
                ):
                    raise CorruptBuildError(
                        "superseded document terminal state is inconsistent"
                    )
                return artifact, run, None, None, True
            if run.status == GraphBuildRun.Status.SUCCEEDED:
                raise CorruptBuildError("successful document run is not terminal")
            live_lease = (
                run.lease_owner
                and run.lease_expires_at is not None
                and run.lease_expires_at > timezone.now()
            )
            if live_lease:
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
            artifact = GraphArtifact.objects.create(
                status=GraphArtifact.Status.BUILDING,
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
    from apps.knowledge_graph.extraction.pipeline import extraction_commit_is_valid
    from apps.knowledge_graph.models import (
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        RelationMention,
    )
    from apps.knowledge_graph.resolution.persistence import (
        resolution_commit_is_valid,
        source_mention_fingerprint,
    )

    mentions = tuple(EntityMention.objects.filter(artifact=artifact).order_by("pk"))
    relation_count = RelationMention.objects.filter(artifact=artifact).count()
    if not extraction_commit_is_valid(
        run,
        entity_count=len(mentions),
        relation_count=relation_count,
    ):
        raise CorruptBuildError("document extraction commit is incomplete")
    entity_count = DocumentEntity.objects.filter(artifact=artifact).count()
    membership_count = DocumentEntityMention.objects.filter(
        document_entity__artifact=artifact
    ).count()
    stats = _safe_marker(run.stats)
    marker = stats.get("resolution_commit")
    marker_map = marker if type(marker) is dict else {}
    fingerprint = source_mention_fingerprint(mentions)
    result_checksum = marker_map.get("result_checksum")
    if not resolution_commit_is_valid(
        marker,
        resolver_version=artifact.resolver_version,
        ontology_checksum=artifact.ontology_checksum,
        assembly_version=artifact.assembly_version,
        assembly_config_checksum=artifact.assembly_config_checksum,
        source_mention_count=len(mentions),
        source_mention_fingerprint=fingerprint,
        document_entity_count=entity_count,
        membership_count=membership_count,
        result_checksum=result_checksum,
    ):
        raise CorruptBuildError("document resolution commit is incomplete")
    return {
        "entity_mention_count": len(mentions),
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
    sequence = list(sequence) if type(sequence) is list else []
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
            context.identity.document_id
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
            or run.stage != GraphBuildRun.Stage.VALIDATING
        ):
            raise CorruptBuildError("document candidate is not validating exact key")
        counts = _document_commit_counts(artifact, run)
        active = tuple(
            row
            for row in artifacts
            if row.status == GraphArtifact.Status.ACTIVE and row.pk != artifact.pk
        )
        if len(active) > 1:
            raise CorruptBuildError("document scope has multiple active artifacts")
        if active and active[0].pk > artifact.pk:
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
            return active[0], counts
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
        artifacts, runs, _document, _chunks = _lock_document_build_rows(
            context.identity.document_id
        )
        artifact = next((row for row in artifacts if row.pk == artifact_id), None)
        run = next((row for row in runs if row.pk == run_id), None)
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
    from apps.knowledge_graph.resolution.coreference import resolve_document_mentions
    from apps.knowledge_graph.resolution.persistence import persist_document_resolution

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
            if not _commit_marker_present(run, "extraction_commit"):
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
            if not _commit_marker_present(run, "resolution_commit"):
                mentions = tuple(
                    EntityMention.objects.filter(artifact=artifact).order_by("pk")
                )
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
        error_code = "source_or_config_stale" if stale else "document_build_failed"
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
        except BuildLeaseLostError:
            pass
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
    from apps.documents.models import Document
    from apps.knowledge_graph.extraction.pipeline import (
        StaleSourceError,
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
    from apps.knowledge_graph.models import GraphArtifact
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
    embedding_model_signature = (
        strict_index_embedding_signature()
        if embedding_model_signature is None
        else embedding_model_signature
    )
    documents = tuple(Document.filter(collection_id=collection_id))
    document_ids = tuple(document.id for document in documents)
    if len(document_ids) != len(set(document_ids)):
        raise CorruptBuildError("collection contains duplicate concrete document UUIDs")
    documents_by_id = {str(document.id): document for document in documents}
    artifacts = tuple(
        GraphArtifact.objects.filter(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id__in=tuple(documents_by_id),
            status=GraphArtifact.Status.ACTIVE,
            ontology_version=ontology.version,
            ontology_checksum=ontology.checksum,
        ).order_by("pk")
    )
    contributing_rows = []
    for artifact in artifacts:
        document = documents_by_id[artifact.scope_id]
        metadata = artifact.metadata if type(artifact.metadata) is dict else {}
        current_chunks = tuple(
            document.chunks.only(
                "pk",
                "doc_id",
                "chunk_number",
                "start_position",
                "end_position",
                "modality",
                "content",
            ).order_by("chunk_number", "pk")
        )
        current_chunk_signature = ordered_chunk_signature(
            current_chunks,
            concrete_model_label=document._meta.label_lower,
        )
        try:
            _validate_source(document, artifact.source_hash)
            source_matches = True
        except StaleSourceError:
            source_matches = False
        if source_matches and (
            metadata.get("orchestration_version") != 1
            or (
                metadata.get("ordered_chunk_signature") == current_chunk_signature
                and metadata.get("ontology_activation_signature")
                == ontology_activation_signature
            )
        ):
            contributing_rows.append(artifact)
    contributing = tuple(contributing_rows)
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


def _lock_collection_build_rows(collection_id: int):
    from apps.knowledge_graph.graph.assembly import lock_collection_graph_scope
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        GraphArtifact,
        GraphBuildRun,
    )

    collection = lock_collection_graph_scope(collection_id)
    artifacts = tuple(
        GraphArtifact.objects.select_for_update()
        .filter(
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            scope_id=str(collection_id),
        )
        .order_by("pk")
    )
    runs = tuple(
        GraphBuildRun.objects.select_for_update()
        .filter(
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            scope_id=str(collection_id),
        )
        .order_by("pk")
    )
    manifests = tuple(
        CollectionArtifactInput.objects.select_for_update()
        .filter(
            artifact__scope_type=GraphArtifact.ScopeType.COLLECTION,
            artifact__scope_id=str(collection_id),
        )
        .order_by("artifact_id", "document_artifact_id")
    )
    return collection, artifacts, runs, manifests


def _bootstrap_collection_build(
    context: _CollectionContext,
    build_key: str,
) -> tuple[object, object, str | None, int | None, bool]:
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.resolution.collection import build_collection_snapshot

    owner = uuid.uuid4().hex
    with transaction.atomic():
        _collection, artifacts, runs, _manifests = _lock_collection_build_rows(
            context.identity.collection_id
        )
        matching_artifacts = tuple(
            row for row in artifacts if row.build_key == build_key
        )
        matching_runs = tuple(row for row in runs if row.build_key == build_key)
        if len(matching_artifacts) > 1 or len(matching_runs) > 1:
            raise CorruptBuildError("collection build key is not unique")
        artifact = matching_artifacts[0] if matching_artifacts else None
        run = matching_runs[0] if matching_runs else None
        if (artifact is None) != (run is None):
            raise CorruptBuildError("collection artifact/run ownership is incomplete")
        if run is not None:
            if run.artifact_id != artifact.pk or run.build_key != artifact.build_key:
                raise CorruptBuildError("collection build ownership is inconsistent")
            if run.stage == GraphBuildRun.Stage.ACTIVE:
                if (
                    run.status != GraphBuildRun.Status.SUCCEEDED
                    or artifact.status != GraphArtifact.Status.ACTIVE
                    or run.lease_owner
                    or run.lease_expires_at is not None
                ):
                    raise CorruptBuildError(
                        "active collection terminal state is inconsistent"
                    )
                return artifact, run, None, None, True
            if run.stage == GraphBuildRun.Stage.SUPERSEDED:
                if (
                    run.status != GraphBuildRun.Status.CANCELLED
                    or artifact.status != GraphArtifact.Status.SUPERSEDED
                    or run.lease_owner
                    or run.lease_expires_at is not None
                ):
                    raise CorruptBuildError(
                        "superseded collection terminal state is inconsistent"
                    )
                return artifact, run, None, None, True
            live_lease = (
                run.lease_owner
                and run.lease_expires_at is not None
                and run.lease_expires_at > timezone.now()
            )
            if live_lease:
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
        lease_owner, lease_generation = _claim_locked_run(run, owner)
        return artifact, run, lease_owner, lease_generation, False


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
        _collection, artifacts, runs, _manifests = _lock_collection_build_rows(
            context.identity.collection_id
        )
        artifact = next((row for row in artifacts if row.pk == artifact_id), None)
        run = next((row for row in runs if row.pk == run_id), None)
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
                )
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
            if not _commit_marker_present(run, "collection_resolution_commit"):
                snapshot, entities, relations = load_collection_resolution_inputs(
                    artifact.pk,
                    run.pk,
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
            run.refresh_from_db()
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
            "collection_identity_changed" if stale else "collection_build_failed"
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
        except BuildLeaseLostError:
            pass
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
    "BuildLeaseLostError",
    "CollectionBuildIdentity",
    "DocumentBuildIdentity",
    "build_document_graph",
    "derive_collection_build_key",
    "derive_document_build_key",
    "refresh_collection_graph",
    "validate_orchestration_stage",
    "validate_build_lease",
    "validate_stage_transition",
]
