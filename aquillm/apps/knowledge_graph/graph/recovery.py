"""Bounded recovery for graph builds whose broker publication was lost."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.stdlib.get_logger(__name__)

_DOCUMENT_PHASE = "documents"
_COLLECTION_PHASE = "collections"
_PHASES = frozenset({_DOCUMENT_PHASE, _COLLECTION_PHASE})
_MAX_PAGE_SIZE = 500


class RecoveryOutcome(StrEnum):
    CURRENT = "current"
    PUBLISHED = "published"
    DEPENDENCY_PENDING = "dependency_pending"
    INVALID = "invalid"
    PUBLISH_FAILED = "publish_failed"


@dataclass(frozen=True, slots=True)
class DocumentRecoveryRow:
    document_id: uuid.UUID
    source_hash: str


@dataclass(frozen=True, slots=True)
class GraphRecoveryCursor:
    phase: str = _DOCUMENT_PHASE
    document_model_index: int = 0
    last_pk: int = 0

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError("graph recovery cursor phase is invalid")
        if type(self.document_model_index) is not int or self.document_model_index < 0:
            raise ValueError("graph recovery document model cursor is invalid")
        if type(self.last_pk) is not int or self.last_pk < 0:
            raise ValueError("graph recovery primary-key cursor is invalid")
        if self.phase == _COLLECTION_PHASE and self.document_model_index != 0:
            raise ValueError("collection recovery cursor cannot name a document model")

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "document_model_index": self.document_model_index,
            "last_pk": self.last_pk,
        }


def _source_hash(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("document recovery source hash must be lowercase SHA-256")
    return value


def _document_id(value: object) -> uuid.UUID:
    if type(value) is not uuid.UUID or value.version is None:
        raise ValueError("document recovery id must be an exact RFC 4122 UUID")
    return value


def _page_size(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_PAGE_SIZE:
        raise ValueError(f"graph recovery page size must be 1..{_MAX_PAGE_SIZE}")
    return value


def _cursor(value: object) -> GraphRecoveryCursor:
    if value is None:
        return GraphRecoveryCursor()
    if type(value) is not dict or set(value) != {
        "phase",
        "document_model_index",
        "last_pk",
    }:
        raise ValueError("graph recovery cursor must be an exact mapping")
    return GraphRecoveryCursor(
        phase=value["phase"],
        document_model_index=value["document_model_index"],
        last_pk=value["last_pk"],
    )


def _document_models() -> tuple[object, ...]:
    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    return tuple(sorted(DESCENDED_FROM_DOCUMENT, key=lambda model: model._meta.label))


def _load_document_page(
    cursor: GraphRecoveryCursor,
    page_size: int,
) -> tuple[tuple[DocumentRecoveryRow, ...], GraphRecoveryCursor]:
    models = _document_models()
    model_index = cursor.document_model_index
    last_pk = cursor.last_pk
    while model_index < len(models):
        rows = tuple(
            models[model_index]
            ._base_manager.filter(
                ingestion_complete=True,
                pk__gt=last_pk,
            )
            .order_by("pk")
            .values("pk", "id", "full_text_hash")[:page_size]
        )
        if rows:
            return (
                tuple(
                    DocumentRecoveryRow(
                        document_id=row["id"],
                        source_hash=row["full_text_hash"],
                    )
                    for row in rows
                ),
                GraphRecoveryCursor(
                    phase=_DOCUMENT_PHASE,
                    document_model_index=model_index,
                    last_pk=int(rows[-1]["pk"]),
                ),
            )
        model_index += 1
        last_pk = 0
    return (), GraphRecoveryCursor(phase=_COLLECTION_PHASE)


def _load_collection_page(
    cursor: GraphRecoveryCursor,
    page_size: int,
) -> tuple[tuple[int, ...], GraphRecoveryCursor | None]:
    from apps.collections.models import Collection

    collection_ids = tuple(
        Collection._base_manager.filter(pk__gt=cursor.last_pk)
        .order_by("pk")
        .values_list("pk", flat=True)[:page_size]
    )
    if not collection_ids:
        return (), None
    return (
        collection_ids,
        GraphRecoveryCursor(
            phase=_COLLECTION_PHASE,
            last_pk=int(collection_ids[-1]),
        ),
    )


def _exact_artifact_exists(*, scope_type: str, scope_id: str, build_key: str) -> bool:
    from apps.knowledge_graph.models import GraphArtifact

    return GraphArtifact.objects.filter(
        scope_type=scope_type,
        scope_id=scope_id,
        status=GraphArtifact.Status.ACTIVE,
        evaluation_only=False,
        build_key=build_key,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
    ).exists()


def _recover_document(
    document_id: uuid.UUID,
    source_hash: str,
) -> RecoveryOutcome:
    from apps.knowledge_graph.extraction.pipeline import StaleSourceError
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.services import builds

    try:
        document_id = _document_id(document_id)
        source_hash = _source_hash(source_hash)
        build_key = builds.derive_current_document_build_key(document_id, source_hash)
    except (LookupError, builds.StaleBuildError):
        return RecoveryOutcome.DEPENDENCY_PENDING
    except (ValueError, builds.CorruptBuildError, StaleSourceError):
        return RecoveryOutcome.INVALID
    if _exact_artifact_exists(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=str(document_id),
        build_key=build_key,
    ):
        return RecoveryOutcome.CURRENT
    try:
        builds.enqueue_document_build(document_id, source_hash)
    except (LookupError, builds.StaleBuildError):
        return RecoveryOutcome.DEPENDENCY_PENDING
    except (ValueError, builds.CorruptBuildError, StaleSourceError):
        return RecoveryOutcome.INVALID
    except Exception as exc:
        logger.error(
            "obs.kg.graph_recovery_publish_failed",
            build_kind="document",
            scope_id=str(document_id),
            error_type=type(exc).__name__,
        )
        return RecoveryOutcome.PUBLISH_FAILED
    return RecoveryOutcome.PUBLISHED


def _recover_collection(collection_id: int) -> RecoveryOutcome:
    from apps.knowledge_graph.extraction.pipeline import StaleSourceError
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.services import builds

    try:
        context = builds._collection_context(collection_id)
        build_key = builds.derive_collection_build_key(context.identity)
    except (LookupError, builds.StaleBuildError):
        return RecoveryOutcome.DEPENDENCY_PENDING
    except (ValueError, builds.CorruptBuildError, StaleSourceError):
        return RecoveryOutcome.INVALID
    if _exact_artifact_exists(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=str(collection_id),
        build_key=build_key,
    ):
        return RecoveryOutcome.CURRENT
    try:
        builds.enqueue_collection_refresh(
            collection_id,
            context.identity.aggregate_source_signature,
            build_key,
        )
    except (LookupError, builds.StaleBuildError):
        return RecoveryOutcome.DEPENDENCY_PENDING
    except (ValueError, builds.CorruptBuildError, StaleSourceError):
        return RecoveryOutcome.INVALID
    except Exception as exc:
        logger.error(
            "obs.kg.graph_recovery_publish_failed",
            build_kind="collection",
            scope_id=str(collection_id),
            error_type=type(exc).__name__,
        )
        return RecoveryOutcome.PUBLISH_FAILED
    return RecoveryOutcome.PUBLISHED


def _summary(
    outcomes: tuple[RecoveryOutcome, ...],
    next_cursor: GraphRecoveryCursor | None,
) -> dict[str, Any]:
    return {
        "examined_count": len(outcomes),
        "current_count": outcomes.count(RecoveryOutcome.CURRENT),
        "published_count": outcomes.count(RecoveryOutcome.PUBLISHED),
        "dependency_pending_count": outcomes.count(RecoveryOutcome.DEPENDENCY_PENDING),
        "invalid_count": outcomes.count(RecoveryOutcome.INVALID),
        "publish_failed_count": outcomes.count(RecoveryOutcome.PUBLISH_FAILED),
        "next_cursor": None if next_cursor is None else next_cursor.as_dict(),
    }


def recover_graph_builds_page(
    cursor: object = None,
    *,
    page_size: int,
) -> dict[str, Any]:
    """Inspect and publish at most one bounded page of exact current scopes."""

    resolved_cursor = _cursor(cursor)
    size = _page_size(page_size)
    if resolved_cursor.phase == _DOCUMENT_PHASE:
        rows, next_cursor = _load_document_page(resolved_cursor, size)
        outcomes = tuple(
            _recover_document(row.document_id, row.source_hash) for row in rows
        )
    else:
        collection_ids, next_cursor = _load_collection_page(resolved_cursor, size)
        outcomes = tuple(
            _recover_collection(collection_id) for collection_id in collection_ids
        )
    return _summary(outcomes, next_cursor)


__all__ = [
    "DocumentRecoveryRow",
    "GraphRecoveryCursor",
    "RecoveryOutcome",
    "recover_graph_builds_page",
]
