"""Prune selection and guarded generation deletion operations."""

from __future__ import annotations

from uuid import UUID

from django.db.models import F, Subquery, Window
from django.db.models.functions import RowNumber
from django.utils import timezone

from apps.knowledge_graph.models import CollectionGraphProjection

from .identifiers import ProjectionIdentifierDomain
from .runtime import ProjectionDatabaseAliases
from .state_repository import FunctionProjectionStateRepository

_ALIASES = ProjectionDatabaseAliases()


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
        return tuple(
            query.filter(pk=projection_id, pruned_at__isnull=True).order_by("id")[
                :page_size
            ]
        )
    if collection_id is not None:
        query = query.filter(collection_pk_snapshot=collection_id)
    # Rank all historical generations before excluding completed deletions;
    # otherwise each pass would retain another group of already old rows.
    eligible = (
        query.annotate(
            generation_rank=Window(
                expression=RowNumber(),
                partition_by=[F("collection_pk_snapshot")],
                order_by=[F("created_at").desc(), F("id").desc()],
            )
        )
        .filter(generation_rank__gt=retain)
        .values("pk")
    )
    return tuple(
        query.filter(pk__in=Subquery(eligible), pruned_at__isnull=True).order_by(
            "collection_pk_snapshot", "-created_at", "id"
        )[:page_size]
    )


def _delete_projection_generation(*, row, graph, settings, codec) -> bool | None:
    if row.state not in {"failed", "superseded"}:
        raise ValueError("only terminal projection authority may be pruned")
    generation_key = codec(
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


def _record_pruned(row) -> None:
    FunctionProjectionStateRepository().record_pruned(
        projection_id=row.id,
        generation_key=row.generation_key,
        now=timezone.now(),
    )


def _prepare_prune(row) -> bool:
    return FunctionProjectionStateRepository().begin_prune(
        projection_id=row.id,
        generation_key=row.generation_key,
        now=timezone.now(),
    )
