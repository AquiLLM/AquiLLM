from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from uuid import UUID

from django.conf import settings
from django.db import transaction
from django.db.models import F, Window
from django.db.models.functions import RowNumber
from django.utils import timezone

from apps.knowledge_graph.models import CollectionGraphProjection, GraphArtifact

from .lifecycle import (
    claim_projection_lease,
    enqueue_collection_projection_locked,
    mark_projection_failed,
    publish_projection_ready_compare_and_set,
)
from .memgraph_driver import Neo4jMemgraphDriver
from .memgraph_repository import MemgraphProjectionRepository
from .postgres_repository import PostgresProjectionRepository
from .records import (
    ProjectionFailureCode,
    ProjectionGenerationManifestV1,
    ProjectionLifecycleState,
)
from .serialization import projection_checksum

_MAX_PAGE = 5_000


@dataclass(frozen=True, slots=True)
class ProjectionRunOutcomeV1:
    projection_id: UUID
    ready: bool
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class ReconcileSummaryV1:
    examined_count: int
    enqueued_count: int
    dry_run: bool


@dataclass(frozen=True, slots=True)
class PruneSummaryV1:
    candidate_count: int
    deleted_count: int
    dry_run: bool


def _size(value: object, name: str) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_PAGE:
        raise ValueError(f"{name} must be an integer in 1..5000")
    return value


def _identifier(value: object) -> UUID:
    if type(value) is not UUID:
        raise TypeError("projection_id must be an exact UUID")
    return value


def _atomic(using: str):
    return transaction.atomic(using=using)


def _postgres_repository():
    return PostgresProjectionRepository()


def _memgraph_repository():
    uri = getattr(settings, "KG_MEMGRAPH_URI", "")
    database = getattr(settings, "KG_MEMGRAPH_DATABASE", "memgraph")
    if not uri:
        raise RuntimeError("memgraph_projection_not_configured")
    driver = Neo4jMemgraphDriver(
        uri,
        getattr(settings, "KG_MEMGRAPH_USERNAME", ""),
        getattr(settings, "KG_MEMGRAPH_PASSWORD", ""),
        database=database,
    )
    return MemgraphProjectionRepository(driver)


def _expected_manifest(bundle, projection_id: UUID):
    checksum = projection_checksum(bundle)
    private = (
        CollectionGraphProjection.objects.only("private_mapping_checksum")
        .get(pk=projection_id)
        .private_mapping_checksum
    )
    return ProjectionGenerationManifestV1(
        bundle.generation.generation_key,
        bundle.generation.schema_version,
        bundle.generation.projection_version,
        bundle.generation.identifier_key_version,
        checksum,
        checksum,
        private,
        bundle.counts,
        ProjectionLifecycleState.BUILDING,
    )


def project_generation(
    *, projection_id: UUID, lease_owner: str
) -> ProjectionRunOutcomeV1:
    identifier = _identifier(projection_id)
    now = timezone.now()
    lease = claim_projection_lease(
        projection_id=identifier,
        owner=lease_owner,
        now=now,
        lease_seconds=300,
        using="default",
    )
    if lease is None:
        return ProjectionRunOutcomeV1(identifier, False, "lease_lost")
    try:
        bundle = _postgres_repository().load_projection_bundle(
            projection_id=identifier, batch_size=1_000
        )
        graph = _memgraph_repository()
        graph.write_staging_generation(
            bundle=bundle, batch_size=1_000, timeout_seconds=5.0
        )
        validation = graph.validate_generation(
            expected=_expected_manifest(bundle, identifier), timeout_seconds=5.0
        )
        if not validation.valid:
            raise ValueError("projection_validation_failed")
        graph.mark_generation_ready(
            generation_key=MemgraphProjectionRepository.opaque_generation_key(
                bundle.generation.generation_key
            ),
            validation_checksum=validation.validation_checksum,
            timeout_seconds=5.0,
        )
        outcome = publish_projection_ready_compare_and_set(
            projection_id=identifier,
            owner=lease_owner,
            validation=validation,
            now=timezone.now(),
            using="default",
        )
        return ProjectionRunOutcomeV1(
            identifier, outcome.published, outcome.failure_code
        )
    except Exception as exc:
        code = (
            ProjectionFailureCode.VALIDATION_FAILED
            if str(exc) == "projection_validation_failed"
            else ProjectionFailureCode.WRITE_FAILED
        )
        try:
            mark_projection_failed(
                projection_id=identifier,
                owner=lease_owner,
                failure_code=code,
                now=timezone.now(),
                using="default",
            )
        except Exception:
            pass
        return ProjectionRunOutcomeV1(identifier, False, code.value)


def _active_artifact_page(*, after_id: int, page_size: int):
    return tuple(
        GraphArtifact.objects.filter(
            pk__gt=after_id,
            scope_type="collection",
            status="active",
            evaluation_only=False,
        )
        .order_by("pk")
        .values_list("collection_scope_id", "pk")[:page_size]
    )


def reconcile_graph_projections(*, page_size: int, dry_run: bool) -> ReconcileSummaryV1:
    size = _size(page_size, "page_size")
    if type(dry_run) is not bool:
        raise TypeError("dry_run must be exact")
    after = examined = enqueued = 0
    while True:
        page = _active_artifact_page(after_id=after, page_size=size)
        if not page:
            break
        for collection_id, artifact_id in page:
            examined += 1
            if not dry_run:
                with _atomic("default"):
                    enqueue_collection_projection_locked(
                        collection_id=collection_id,
                        artifact_id=artifact_id,
                        using="default",
                    )
                enqueued += 1
        after = max(row[1] for row in page)
        if len(page) < size:
            break
    return ReconcileSummaryV1(examined, enqueued, dry_run)


def _prune_candidates(*, page_size: int, retain: int):
    return tuple(
        CollectionGraphProjection.objects.filter(state__in=("failed", "superseded"))
        .annotate(
            generation_rank=Window(
                expression=RowNumber(),
                partition_by=[F("collection_pk_snapshot")],
                order_by=F("created_at").desc(),
            )
        )
        .filter(generation_rank__gt=retain)
        .order_by("collection_pk_snapshot", "-created_at", "id")[:page_size]
    )


def _delete_projection_generation(row) -> None:
    value = sha256(
        f"projection-generation-v1\0{row.generation_key}".encode()
    ).hexdigest()
    graph = _memgraph_repository()
    graph.delete_generation(
        generation_key=graph.opaque_generation_key(value), timeout_seconds=5.0
    )


def prune_graph_projection_generations(
    *, page_size: int, retain: int, dry_run: bool
) -> PruneSummaryV1:
    size = _size(page_size, "page_size")
    if type(retain) is not int or not 1 <= retain <= _MAX_PAGE:
        raise ValueError("retain must be an integer in 1..5000")
    if type(dry_run) is not bool:
        raise TypeError("dry_run must be exact")
    candidates = _prune_candidates(page_size=size, retain=retain)
    deleted = 0
    if not dry_run:
        for row in candidates:
            _delete_projection_generation(row)
            deleted += 1
    return PruneSummaryV1(len(candidates), deleted, dry_run)


def inspect_projection_authority(
    *, collection_id: int | None, all_collections: bool, page_size: int
) -> dict[str, int]:
    _size(page_size, "page_size")
    query = CollectionGraphProjection.objects.all()
    if not all_collections:
        if type(collection_id) is not int or collection_id < 1:
            raise ValueError("collection_id must be positive")
        query = query.filter(collection_pk_snapshot=collection_id)
    counts = {"ready_count": 0, "pending_count": 0, "failed_count": 0}
    for state in query.order_by("id").values_list("state", flat=True)[:page_size]:
        key = f"{state}_count"
        if key in counts:
            counts[key] += 1
    counts["drift_count"] = counts["failed_count"]
    return counts
