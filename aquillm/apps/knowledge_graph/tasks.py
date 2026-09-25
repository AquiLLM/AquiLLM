"""Thin Celery boundaries for isolated knowledge-graph work.

Provider and build-service imports stay inside task execution so ordinary
workers can register these tasks without the optional ML runtime installed.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import structlog
from celery import shared_task
from django.conf import settings

from apps.knowledge_graph.projection import tasks as projection_tasks  # noqa: F401
from lib.knowledge_graph.config import (
    KnowledgeGraphConfigError,
    get_build_enabled,
    validate_extraction_queue,
)

from .task_retry import (
    _retry_countdown,
)

GRAPH_EXTRACTION_QUEUE = settings.KG_EXTRACTION_QUEUE
MAX_BUILD_RETRIES = 3
_INVALID_SCOPE_ID = "invalid"
_LOWER_HEX_DIGITS = frozenset("0123456789abcdef")


logger = structlog.stdlib.get_logger(__name__)


def _task_extraction_queue_is_valid() -> bool:
    if getattr(settings, "KG_EXTRACTION_QUEUE_VALID", False) is not True:
        return False
    try:
        configured_queue = validate_extraction_queue(settings.KG_EXTRACTION_QUEUE)
    except KnowledgeGraphConfigError:
        return False
    return configured_queue == GRAPH_EXTRACTION_QUEUE


def _canonical_document_id(value: object) -> uuid.UUID:
    if type(value) is not str:
        raise ValueError("document_id must be a canonical UUID string")
    try:
        document_id = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("document_id must be a canonical UUID string") from exc
    if str(document_id) != value:
        raise ValueError("document_id must be a canonical UUID string")
    return document_id


def _canonical_request_id(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError("request_id must be a canonical UUID string")
    try:
        request_id = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("request_id must be a canonical UUID string") from exc
    if str(request_id) != value:
        raise ValueError("request_id must be a canonical UUID string")
    return value


def _assert_task_evaluation_bypass(eval_only: object) -> bool:
    """Re-read Django settings at the asynchronous trust boundary."""

    if type(eval_only) is not bool:
        raise ValueError("evaluation marker must be an exact boolean")
    if not eval_only:
        return False
    from django.conf import settings

    if not (
        getattr(settings, "KG_EVAL_BYPASS_ALLOWED", False) is True
        and (
            getattr(settings, "DEBUG", False) is True
            or getattr(settings, "TESTING", False) is True
        )
    ):
        raise PermissionError(
            "evaluation-only graph bypass is not authorized in this environment"
        )
    return True


def _collection_id(value: object) -> int:
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError("collection_id must be a positive database integer")
    return value


def _lower_hex_hash(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{field} must be exactly 64 lowercase hexadecimal characters")
    return value




def _record_terminal_task_failure(
    *,
    task_name: str,
    build_kind: str,
    scope_id: str,
    exc: BaseException,
    retry_count: int,
) -> None:
    logger.error(
        "obs.kg.task_failed",
        task_name=task_name,
        build_kind=build_kind,
        scope_id=scope_id,
        error_type=type(exc).__name__,
        retry_count=retry_count,
        terminal=True,
    )


def _record_request_failure_if_exact(
    *,
    request_id: str,
    eval_only: bool,
    build_kind: str,
    scope_id: uuid.UUID | int,
    source_hash: str,
    error_code: str,
) -> bool:
    """Mutate only the request exactly named by canonical task metadata."""

    from apps.knowledge_graph.services import builds

    try:
        request = builds.validate_rebuild_task_request_metadata(
            request_id,
            eval_only,
            build_kind=build_kind,
            scope_id=scope_id,
            source_hash=source_hash,
        )
    except (builds.CorruptBuildError, builds.StaleBuildError, ValueError):
        return False
    if request is None:
        return False
    builds.record_rebuild_failure(request_id, error_code=error_code)
    return True


def _validate_exact_task_request(
    *,
    builds: Any,
    request_id: str | None,
    eval_only: bool,
    build_kind: str,
    scope_id: uuid.UUID | int,
    source_hash: str,
) -> bool:
    if request_id is None:
        return False
    builds.validate_rebuild_task_request_metadata(
        request_id,
        eval_only,
        build_kind=build_kind,
        scope_id=scope_id,
        source_hash=source_hash,
    )
    return True


def _run_build_task(
    task,
    operation: Callable[[], object],
    *,
    builds: Any | None,
    build_kind: str,
    scope_id: str,
    rebuild_request_id: str | None = None,
    rebuild_request_exact: bool = False,
) -> int | None:
    try:
        artifact = operation()
        artifact_id = getattr(artifact, "pk", None)
        if type(artifact_id) is not int or artifact_id <= 0:
            raise ValueError("graph build service returned an invalid artifact id")
        return artifact_id
    except Exception as exc:
        if builds is not None and isinstance(exc, builds.StaleBuildError):
            if rebuild_request_exact:
                builds.record_rebuild_failure(
                    rebuild_request_id,
                    error_code="source_or_config_stale",
                    resnapshot=True,
                )
            return None
        retry_count = int(getattr(task.request, "retries", 0))
        countdown = _retry_countdown(
            exc,
            builds=builds,
            retry_count=retry_count,
        )
        if countdown is not None and retry_count < task.max_retries:
            raise task.retry(exc=exc, countdown=countdown)
        _record_terminal_task_failure(
            task_name=task.name,
            build_kind=build_kind,
            scope_id=scope_id,
            exc=exc,
            retry_count=retry_count,
        )
        if builds is not None and rebuild_request_exact:
            builds.record_rebuild_failure(
                rebuild_request_id,
                error_code="task_terminal_failure",
            )
        raise


@shared_task(
    bind=True,
    name="apps.knowledge_graph.tasks.build_document_graph_task",
    queue=GRAPH_EXTRACTION_QUEUE,
    serializer="json",
    max_retries=MAX_BUILD_RETRIES,
    acks_late=True,
    reject_on_worker_lost=True,
    ignore_result=True,
)
def build_document_graph_task(
    self,
    document_id: str,
    expected_source_hash: str,
    document_build_key: str,
    request_id: str | None = None,
    eval_only: bool = False,
) -> int | None:
    """Build one exact document snapshot from a JSON-safe request."""

    if not _task_extraction_queue_is_valid():
        return None
    scope_id = _INVALID_SCOPE_ID
    resolved_request_id = None
    resolved_document_id = None
    resolved_source_hash = None
    evaluation_checked = False
    try:
        resolved_request_id = _canonical_request_id(request_id)
        resolved_document_id = _canonical_document_id(document_id)
        scope_id = str(resolved_document_id)
        resolved_source_hash = _lower_hex_hash(
            expected_source_hash,
            field="expected_source_hash",
        )
        resolved_build_key = _lower_hex_hash(
            document_build_key,
            field="document_build_key",
        )
        evaluation_authorized = _assert_task_evaluation_bypass(eval_only)
        evaluation_checked = True
        if evaluation_authorized and resolved_request_id is None:
            raise ValueError("evaluation-only task requires a rebuild request id")
        if not get_build_enabled() and not evaluation_authorized:
            if resolved_request_id is not None:
                _record_request_failure_if_exact(
                    request_id=resolved_request_id,
                    eval_only=eval_only,
                    build_kind="document",
                    scope_id=resolved_document_id,
                    source_hash=resolved_source_hash,
                    error_code="task_build_disabled",
                )
            return None
    except Exception as exc:
        _record_terminal_task_failure(
            task_name=self.name,
            build_kind="document",
            scope_id=scope_id,
            exc=exc,
            retry_count=0,
        )
        if (
            resolved_request_id is not None
            and evaluation_checked
            and type(eval_only) is bool
            and resolved_document_id is not None
            and resolved_source_hash is not None
        ):
            _record_request_failure_if_exact(
                request_id=resolved_request_id,
                eval_only=eval_only,
                build_kind="document",
                scope_id=resolved_document_id,
                source_hash=resolved_source_hash,
                error_code="task_payload_invalid",
            )
        raise
    from apps.knowledge_graph.services import builds

    request_is_exact = _validate_exact_task_request(
        builds=builds,
        request_id=resolved_request_id,
        eval_only=eval_only,
        build_kind="document",
        scope_id=resolved_document_id,
        source_hash=resolved_source_hash,
    )

    def build():
        if resolved_request_id is None and not eval_only:
            return builds.build_document_graph(
                resolved_document_id,
                resolved_source_hash,
                resolved_build_key,
            )
        return builds.build_document_graph(
            resolved_document_id,
            resolved_source_hash,
            resolved_build_key,
            request_id=resolved_request_id,
            eval_only=eval_only,
        )

    return _run_build_task(
        self,
        build,
        builds=builds,
        build_kind="document",
        scope_id=str(resolved_document_id),
        rebuild_request_id=resolved_request_id,
        rebuild_request_exact=request_is_exact,
    )


@shared_task(
    bind=True,
    name="apps.knowledge_graph.tasks.refresh_collection_graph_task",
    queue=GRAPH_EXTRACTION_QUEUE,
    serializer="json",
    max_retries=MAX_BUILD_RETRIES,
    acks_late=True,
    reject_on_worker_lost=True,
    ignore_result=True,
)
def refresh_collection_graph_task(
    self,
    collection_id: int,
    aggregate_source_signature: str,
    collection_build_key: str,
    request_id: str | None = None,
    eval_only: bool = False,
) -> int | None:
    """Refresh one exact collection snapshot from JSON-safe values."""

    if not _task_extraction_queue_is_valid():
        return None
    scope_id = _INVALID_SCOPE_ID
    resolved_request_id = None
    resolved_collection_id = None
    resolved_aggregate_signature = None
    evaluation_checked = False
    try:
        resolved_request_id = _canonical_request_id(request_id)
        resolved_collection_id = _collection_id(collection_id)
        scope_id = str(resolved_collection_id)
        resolved_aggregate_signature = _lower_hex_hash(
            aggregate_source_signature,
            field="aggregate_source_signature",
        )
        resolved_build_key = _lower_hex_hash(
            collection_build_key,
            field="collection_build_key",
        )
        evaluation_authorized = _assert_task_evaluation_bypass(eval_only)
        evaluation_checked = True
        if evaluation_authorized and resolved_request_id is None:
            raise ValueError("evaluation-only task requires a rebuild request id")
        if not get_build_enabled() and not evaluation_authorized:
            if resolved_request_id is not None:
                _record_request_failure_if_exact(
                    request_id=resolved_request_id,
                    eval_only=eval_only,
                    build_kind="collection",
                    scope_id=resolved_collection_id,
                    source_hash=resolved_aggregate_signature,
                    error_code="task_build_disabled",
                )
            return None
    except Exception as exc:
        _record_terminal_task_failure(
            task_name=self.name,
            build_kind="collection",
            scope_id=scope_id,
            exc=exc,
            retry_count=0,
        )
        if (
            resolved_request_id is not None
            and evaluation_checked
            and type(eval_only) is bool
            and resolved_collection_id is not None
            and resolved_aggregate_signature is not None
        ):
            _record_request_failure_if_exact(
                request_id=resolved_request_id,
                eval_only=eval_only,
                build_kind="collection",
                scope_id=resolved_collection_id,
                source_hash=resolved_aggregate_signature,
                error_code="task_payload_invalid",
            )
        raise
    from apps.knowledge_graph.services import builds

    request_is_exact = _validate_exact_task_request(
        builds=builds,
        request_id=resolved_request_id,
        eval_only=eval_only,
        build_kind="collection",
        scope_id=resolved_collection_id,
        source_hash=resolved_aggregate_signature,
    )

    def refresh():
        if resolved_request_id is None and not eval_only:
            return builds.refresh_collection_graph(
                resolved_collection_id,
                resolved_aggregate_signature,
                resolved_build_key,
            )
        return builds.refresh_collection_graph(
            resolved_collection_id,
            resolved_aggregate_signature,
            resolved_build_key,
            request_id=resolved_request_id,
            eval_only=eval_only,
        )

    return _run_build_task(
        self,
        refresh,
        builds=builds,
        build_kind="collection",
        scope_id=str(resolved_collection_id),
        rebuild_request_id=resolved_request_id,
        rebuild_request_exact=request_is_exact,
    )


# The configured Redis transport consumes priority 0 before 9, so 9 is the
# maintenance/lowest-priority lane for this deployment.
@shared_task(
    name="apps.knowledge_graph.tasks.prune_graph_artifacts_task",
    queue=GRAPH_EXTRACTION_QUEUE,
    priority=9,
    serializer="json",
    acks_late=True,
    reject_on_worker_lost=True,
    ignore_result=True,
)
def prune_graph_artifacts_task():
    """Run Task 18's pruning service; intentionally absent from beat schedules."""

    if not _task_extraction_queue_is_valid():
        return None
    from apps.knowledge_graph.services.pruning import prune_graph_artifacts

    return prune_graph_artifacts(execute=True)


@shared_task(
    bind=True,
    name="apps.knowledge_graph.tasks.recover_missing_graph_builds_task",
    queue=GRAPH_EXTRACTION_QUEUE,
    priority=9,
    serializer="json",
    acks_late=True,
    reject_on_worker_lost=True,
    ignore_result=True,
)
def recover_missing_graph_builds_task(
    self,
    cursor: dict[str, object] | None = None,
    page_size: int | None = None,
):
    """Recover one bounded page and publish a single continuation."""

    if (
        not _task_extraction_queue_is_valid()
        or getattr(settings, "KG_MAINTENANCE_SCHEDULER_ENABLED", False) is not True
        or not get_build_enabled()
    ):
        return None
    from apps.knowledge_graph.graph.recovery import recover_graph_builds_page

    size = (
        getattr(settings, "KG_GRAPH_RECOVERY_PAGE_SIZE", 50)
        if page_size is None
        else page_size
    )
    summary = recover_graph_builds_page(cursor, page_size=size)
    next_cursor = summary["next_cursor"]
    if next_cursor is not None:
        try:
            self.apply_async(
                kwargs={"cursor": next_cursor, "page_size": size},
                countdown=1,
                queue=GRAPH_EXTRACTION_QUEUE,
                priority=9,
            )
        except Exception as exc:
            logger.error(
                "obs.kg.graph_recovery_continuation_failed",
                error_type=type(exc).__name__,
            )
    return summary


__all__ = [
    "build_document_graph_task",
    "prune_graph_artifacts_task",
    "recover_missing_graph_builds_task",
    "refresh_collection_graph_task",
]
