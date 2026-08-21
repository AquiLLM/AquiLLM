# ruff: noqa: I001
from __future__ import annotations

from uuid import UUID

from django.db import transaction
from django.db.models import F, Window
from django.db.models.functions import RowNumber
from django.utils import timezone

from apps.knowledge_graph.models import CollectionGraphProjection, GraphArtifact

from .generation_audit import audit_projection_generation as _generation_audit, orphan_generation_keys as _orphan_generation_keys  # noqa: E501
from .identifiers import OpaqueProjectionKey, ProjectionIdentifierDomain
from .inspection import inspect_projection_authority as inspect_projection_authority
from .reconciliation_types import PruneSummaryV1, ReconcileSummaryV1
from .runtime import (
    ProjectionDatabaseAliases,
    load_projection_runtime_settings,
    memgraph_projection_repository,
    postgres_projection_repository,
    projection_identifier_codec,
)
from .state_repository import FunctionProjectionStateRepository

_MAX_PAGE, _ALIASES = 5_000, ProjectionDatabaseAliases()


def _size(value: object, name: str) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_PAGE:
        raise ValueError(f"{name} must be an integer in 1..5000")
    return value


def _collection(value: object) -> int | None:
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("collection_id must be a positive integer")
    return value


def _projection_identifier(value: object) -> UUID | None:
    if value is not None and type(value) is not UUID:
        raise TypeError("projection_id must be an exact UUID")
    return value


def _atomic(using: str):
    return transaction.atomic(using=using)


_projection_settings = load_projection_runtime_settings


def _memgraph_repository():
    return memgraph_projection_repository(_projection_settings())


def supersede_projection_locked(*, projection_id, now, using):
    del using
    return FunctionProjectionStateRepository().supersede(
        projection_id=projection_id, now=now
    )


def enqueue_collection_projection_locked(*, collection_id, artifact_id, using, codec):
    del using, codec
    settings = _projection_settings()
    return FunctionProjectionStateRepository().replay(
        projection_id=None, collection_id=collection_id, artifact_id=artifact_id,
        versions=(
            settings.projection_schema_version, settings.projection_format_version,
            settings.projection_identifier_key_version,
        ),
        now=timezone.now(),
    )


def _postgres_repository():
    return postgres_projection_repository(
        _projection_settings(), state_repository=FunctionProjectionStateRepository()
    )


def _active_artifact_page(
    *, after_id: int, page_size: int, collection_id: int | None = None
):
    query = GraphArtifact.objects.using(_ALIASES.source).filter(
        pk__gt=after_id,
        scope_type="collection",
        status="active",
        evaluation_only=False,
    )
    if collection_id is not None:
        query = query.filter(collection_scope_id=collection_id)
    return tuple(
        query.order_by("pk").values_list("collection_scope_id", "pk")[:page_size]
    )


def _projection_for_active(*, collection_id: int, artifact_id: int):
    return (
        CollectionGraphProjection.objects.using(_ALIASES.source)
        .filter(
            collection_pk_snapshot=collection_id,
            artifact_pk_snapshot=artifact_id,
            state__in=("pending", "building", "ready"),
        )
        .order_by("-created_at", "-id")
        .first()
    )


def _replay_projection(*, row, collection_id: int, artifact_id: int, codec) -> None:
    using = _ALIASES.state
    with _atomic(using):
        if row is not None and row.state == "ready":
            supersede_projection_locked(
                projection_id=row.id,
                now=timezone.now(),
                using=using,
            )
        enqueue_collection_projection_locked(
            collection_id=collection_id,
            artifact_id=artifact_id,
            using=using,
            codec=codec,
        )


def reconcile_graph_projections(
    *, page_size: int, dry_run: bool, collection_id: int | None = None
) -> ReconcileSummaryV1:
    size = _size(page_size, "page_size")
    selected_collection = _collection(collection_id)
    if type(dry_run) is not bool:
        raise TypeError("dry_run must be exact")
    settings = _projection_settings()
    postgres, graph = _postgres_repository(), _memgraph_repository()
    codec = None if dry_run else projection_identifier_codec(settings)
    after = examined = enqueued = drift = replayed = 0
    selected_collection_key = None
    while True:
        page = _active_artifact_page(
            after_id=after,
            page_size=size,
            collection_id=selected_collection,
        )
        if not page:
            break
        for active_collection_id, artifact_id in page:
            examined += 1
            row = _projection_for_active(
                collection_id=active_collection_id,
                artifact_id=artifact_id,
            )
            audit = _generation_audit(
                row=row,
                postgres=postgres,
                graph=graph,
                settings=settings,
            )
            reason = audit.replay_reason
            if selected_collection is not None and getattr(
                audit, "collection_key", None
            ):
                selected_collection_key = _opaque_generation(audit.collection_key)
            if reason is not None:
                replayed += 1
                drift += int(reason in {"authority_drift", "checksum_drift"})
                if not dry_run:
                    _replay_projection(
                        row=row,
                        collection_id=active_collection_id,
                        artifact_id=artifact_id,
                        codec=codec,
                    )
                    enqueued += 1
        after = page[-1][1]
        if len(page) < size:
            break
    orphaned = ()
    if selected_collection is None or selected_collection_key is not None:
        orphaned = _orphan_generation_keys(
            postgres=postgres,
            graph=graph,
            settings=settings,
            limit=size,
            collection_id=selected_collection,
            collection_key=selected_collection_key,
        )
    return ReconcileSummaryV1(
        examined, enqueued, dry_run, drift, len(orphaned), replayed
    )


def _prune_candidates(
    *,
    page_size: int,
    retain: int,
    projection_id: UUID | None,
    collection_id: int | None,
):
    query = CollectionGraphProjection.objects.using(_ALIASES.source).filter(
        state__in=("failed", "superseded")
    )
    if projection_id is not None:
        return tuple(query.filter(pk=projection_id).order_by("id")[:page_size])
    if collection_id is not None:
        query = query.filter(collection_pk_snapshot=collection_id)
    return tuple(
        query.annotate(
            generation_rank=Window(
                expression=RowNumber(),
                partition_by=[F("collection_pk_snapshot")],
                order_by=F("created_at").desc(),
            )
        )
        .filter(generation_rank__gt=retain)
        .order_by("collection_pk_snapshot", "-created_at", "id")[:page_size]
    )


def _opaque_generation(value: str) -> OpaqueProjectionKey:
    return OpaqueProjectionKey(ProjectionIdentifierDomain.COLLECTION, value)


def _delete_projection_generation(*, row, graph, settings) -> bool | None:
    if row.state not in {"failed", "superseded"}:
        raise ValueError("only terminal projection authority may be pruned")
    generation_key = projection_identifier_codec(
        settings,
        key_version=row.identifier_key_version,
    ).encode(
        ProjectionIdentifierDomain.COLLECTION,
        generation=row.generation_key,
        source=row.generation_key,
    )
    return graph.delete_generation(
        generation_key=generation_key,
        timeout_seconds=settings.graph_overall_timeout_ms / 1_000.0,
    )


def prune_graph_projection_generations(
    *,
    page_size: int,
    retain: int,
    dry_run: bool,
    projection_id: UUID | None = None,
    collection_id: int | None = None,
) -> PruneSummaryV1:
    size = _size(page_size, "page_size")
    identifier = _projection_identifier(projection_id)
    selected_collection = _collection(collection_id)
    if identifier is not None and selected_collection is not None:
        raise ValueError("projection_id and collection_id are mutually exclusive")
    if type(retain) is not int or not 1 <= retain <= _MAX_PAGE:
        raise ValueError("retain must be an integer in 1..5000")
    if type(dry_run) is not bool:
        raise TypeError("dry_run must be exact")
    settings = _projection_settings()
    graph = _memgraph_repository()
    candidates = _prune_candidates(
        page_size=size,
        retain=retain,
        projection_id=identifier,
        collection_id=selected_collection,
    )
    orphaned = ()
    if identifier is None and selected_collection is None and len(candidates) < size:
        orphaned = _orphan_generation_keys(
            postgres=_postgres_repository(),
            graph=graph,
            settings=settings,
            limit=size - len(candidates),
        )
    deleted = 0
    if not dry_run:
        for row in candidates:
            deleted += int(
                _delete_projection_generation(
                    row=row,
                    graph=graph,
                    settings=settings,
                )
                is not False
            )
        for generation_key in orphaned:
            deleted += int(
                graph.delete_generation(
                    generation_key=generation_key,
                    timeout_seconds=settings.graph_overall_timeout_ms / 1_000.0,
                )
                is not False
            )
    return PruneSummaryV1(
        len(candidates) + len(orphaned), deleted, dry_run, len(orphaned)
    )
