"""Thin Celery boundaries for isolated knowledge-graph work.

Provider and build-service imports stay inside task execution so Django, web,
and ordinary Celery workers can register these tasks without the optional ML
runtime installed.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from typing import Any

import structlog
from celery import shared_task
from django.db import InterfaceError as DjangoInterfaceError
from django.db import OperationalError as DjangoOperationalError
from kombu.exceptions import OperationalError as KombuOperationalError

from lib.knowledge_graph.config import get_build_enabled

GRAPH_EXTRACTION_QUEUE = "knowledge-graph-extraction"
MAX_BUILD_RETRIES = 3
_TRANSIENT_RETRY_BASE_SECONDS = 30
_TRANSIENT_RETRY_MAX_SECONDS = 60
_MAX_EXCEPTION_CHAIN_DEPTH = 16
_INVALID_SCOPE_ID = "invalid"
_LOWER_HEX_DIGITS = frozenset("0123456789abcdef")

_OPENAI_TRANSIENT_ERRORS = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }
)
_BOTOCORE_TRANSIENT_ERRORS = frozenset(
    {
        "ConnectionClosedError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "HTTPClientError",
        "ReadTimeoutError",
    }
)
_BOTOCORE_TRANSIENT_ERROR_CODES = frozenset(
    {
        "internalerror",
        "requestlimitexceeded",
        "requesttimeout",
        "requesttimeoutexception",
        "serviceunavailable",
        "slowdown",
    }
)

logger = structlog.stdlib.get_logger(__name__)


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


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_MAX_EXCEPTION_CHAIN_DEPTH):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            return
        else:
            current = current.__context__


def _response_status_code(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if type(status_code) is int:
        return status_code
    response = getattr(exc, "response", None)
    if type(response) is not dict:
        return None
    metadata = response.get("ResponseMetadata")
    if type(metadata) is not dict:
        return None
    status_code = metadata.get("HTTPStatusCode")
    return status_code if type(status_code) is int else None


def _is_known_provider_transient(exc: BaseException) -> bool:
    exception_type = type(exc)
    if (
        exception_type.__module__ == "lib.knowledge_graph.extractors.gliner2_local"
        and exception_type.__name__ == "ExtractionBackendError"
    ):
        return True
    module_root = exception_type.__module__.partition(".")[0]
    name = exception_type.__name__
    status_code = _response_status_code(exc)
    if module_root == "openai":
        return name in _OPENAI_TRANSIENT_ERRORS or (
            name == "APIStatusError"
            and status_code is not None
            and (status_code == 429 or status_code >= 500)
        )
    if module_root != "botocore":
        return False
    if name in _BOTOCORE_TRANSIENT_ERRORS:
        return True
    if name != "ClientError":
        return False
    response = getattr(exc, "response", None)
    error = response.get("Error") if type(response) is dict else None
    error_code = error.get("Code") if type(error) is dict else None
    return (status_code is not None and (status_code == 429 or status_code >= 500)) or (
        type(error_code) is str
        and (
            error_code.lower().startswith("throttl")
            or error_code.lower() in _BOTOCORE_TRANSIENT_ERROR_CODES
        )
    )


def _retry_countdown(
    exc: BaseException,
    *,
    builds: Any | None,
    retry_count: int,
) -> int | None:
    if builds is not None and isinstance(
        exc,
        (builds.StaleBuildError, builds.CorruptBuildError),
    ):
        return None
    if builds is not None and isinstance(
        exc,
        builds.BuildInProgressError,
    ):
        return builds.BUILD_LEASE_RETRY_SECONDS
    if builds is not None and isinstance(exc, builds.BuildLeaseLostError):
        return min(
            _TRANSIENT_RETRY_BASE_SECONDS * (2**retry_count),
            _TRANSIENT_RETRY_MAX_SECONDS,
        )
    for candidate in _exception_chain(exc):
        if isinstance(
            candidate,
            (
                ConnectionError,
                TimeoutError,
                DjangoInterfaceError,
                DjangoOperationalError,
                KombuOperationalError,
            ),
        ) or _is_known_provider_transient(candidate):
            return min(
                _TRANSIENT_RETRY_BASE_SECONDS * (2**retry_count),
                _TRANSIENT_RETRY_MAX_SECONDS,
            )
    return None


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


def _run_build_task(
    task,
    operation: Callable[[], object],
    *,
    builds: Any | None,
    build_kind: str,
    scope_id: str,
) -> int | None:
    try:
        artifact = operation()
        artifact_id = getattr(artifact, "pk", None)
        if type(artifact_id) is not int or artifact_id <= 0:
            raise ValueError("graph build service returned an invalid artifact id")
        return artifact_id
    except Exception as exc:
        if builds is not None and isinstance(exc, builds.StaleBuildError):
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
) -> int | None:
    """Build one exact document snapshot from a JSON-safe request."""

    scope_id = _INVALID_SCOPE_ID
    try:
        if not get_build_enabled():
            return None
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
    except Exception as exc:
        _record_terminal_task_failure(
            task_name=self.name,
            build_kind="document",
            scope_id=scope_id,
            exc=exc,
            retry_count=0,
        )
        raise
    from apps.knowledge_graph.services import builds

    def build():
        return builds.build_document_graph(
            resolved_document_id,
            resolved_source_hash,
            resolved_build_key,
        )

    return _run_build_task(
        self,
        build,
        builds=builds,
        build_kind="document",
        scope_id=str(resolved_document_id),
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
) -> int | None:
    """Refresh one exact collection snapshot from JSON-safe values."""

    scope_id = _INVALID_SCOPE_ID
    try:
        if not get_build_enabled():
            return None
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
    except Exception as exc:
        _record_terminal_task_failure(
            task_name=self.name,
            build_kind="collection",
            scope_id=scope_id,
            exc=exc,
            retry_count=0,
        )
        raise
    from apps.knowledge_graph.services import builds

    def refresh():
        return builds.refresh_collection_graph(
            resolved_collection_id,
            resolved_aggregate_signature,
            resolved_build_key,
        )

    return _run_build_task(
        self,
        refresh,
        builds=builds,
        build_kind="collection",
        scope_id=str(resolved_collection_id),
    )


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

    if not get_build_enabled():
        return None
    from apps.knowledge_graph.services.pruning import prune_graph_artifacts

    return prune_graph_artifacts()


__all__ = [
    "build_document_graph_task",
    "prune_graph_artifacts_task",
    "refresh_collection_graph_task",
]
