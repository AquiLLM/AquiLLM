from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from django.db import transaction
from django.db.models import F, Window
from django.db.models.functions import RowNumber

from apps.knowledge_graph.models import CollectionGraphProjection, GraphArtifact

from .lifecycle import enqueue_collection_projection_locked
from .runtime import (
    load_projection_runtime_settings,
    memgraph_projection_repository,
)

_MAX_PAGE = 5_000


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


def _atomic(using: str):
    return transaction.atomic(using=using)


def _projection_settings():
    return load_projection_runtime_settings()


def _memgraph_repository():
    return memgraph_projection_repository(_projection_settings())


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
