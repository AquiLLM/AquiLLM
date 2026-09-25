# ruff: noqa: E402, I001 - worker imports lease helpers after task setup
"""Durable, local-only Celery boundary for collection schema generation."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import timedelta

import structlog
from celery import shared_task
from django.db import transaction
from django.utils import timezone

from apps.collections.services.schema_generation import (
    InvalidSchemaCandidate,
    _locked_collection_source_signature as _locked_collection_source_signature,
    collect_candidate_evidence,
    collection_ingestion_pending as collection_ingestion_pending,
    generate_schema_candidate,
    load_schema_generation_config,
    sample_collection_chunks,
)

logger = structlog.stdlib.get_logger(__name__)
_MAX_RETRIES = 3
_RETRY_COUNTDOWN_SECONDS = 30
_LEASE_DURATION = timedelta(minutes=10)
_QUEUE = "knowledge-graph-schema"
_MAX_SOURCE_DEFERRALS = 20
_SOURCE_SETTLING_DURATION = timedelta(minutes=10)


class _SourceChanged(RuntimeError):
    """Uploads are still active or the attempt's source snapshot changed."""


class _LeaseLost(RuntimeError):
    """A newer delivery owns the run, so this stale delivery must do nothing."""


class _LeaseBusy(RuntimeError):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class _RunClaim:
    run: object
    lease_token: uuid.UUID


def _canonical_run_id(value: object) -> uuid.UUID:
    if type(value) is not str:
        raise ValueError("run_id must be a canonical UUID string")
    parsed = uuid.UUID(value)
    if str(parsed) != value:
        raise ValueError("run_id must be a canonical UUID string")
    return parsed


def _generation_enabled() -> bool:
    return (os.environ.get("KG_SCHEMA_GENERATION_ENABLED") or "0").strip() == "1"


def enqueue_schema_generation(run_id) -> None:
    """Publish an exact durable run identifier to the isolated schema queue."""

    parsed = run_id if isinstance(run_id, uuid.UUID) else _canonical_run_id(run_id)
    generate_collection_schema_task.delay(str(parsed))


def _fail_run(run_id: uuid.UUID, error_code: str, lease_token: uuid.UUID) -> bool:
    from apps.collections.models import CollectionSchemaGenerationRun

    filters = {
        "id": run_id,
        "status__in": ("queued", "running"),
        "lease_token": lease_token,
        "lease_expires_at__gt": timezone.now(),
    }
    return bool(
        CollectionSchemaGenerationRun.objects.filter(**filters).update(
            status="failed",
            error_code=error_code,
            completed_at=timezone.now(),
            lease_token=None,
            lease_expires_at=None,
        )
    )


def _fail_expired_run(
    run_id: uuid.UUID, error_code: str, lease_token: uuid.UUID
) -> bool:
    """Fail only the same running owner after its lease actually expired."""

    from apps.collections.models import CollectionSchemaGenerationRun

    return bool(
        CollectionSchemaGenerationRun.objects.filter(
            id=run_id,
            status="running",
            lease_token=lease_token,
            lease_expires_at__lte=timezone.now(),
        ).update(
            status="failed",
            error_code=error_code,
            completed_at=timezone.now(),
            lease_token=None,
            lease_expires_at=None,
        )
    )


def _safe_log_failure(error_code: str, exc: BaseException | None = None) -> None:
    """Log only a public code and exception class, never source text or secrets."""

    logger.error(
        "obs.collections.schema_generation_failed",
        error_code=error_code,
        error_type=None if exc is None else type(exc).__name__,
    )


def _release_lease_for_retry(run_id: uuid.UUID, lease_token: uuid.UUID) -> bool:
    """Return a failed attempt to queued only when it still owns the run lease."""

    from apps.collections.models import CollectionSchemaGenerationRun

    with transaction.atomic():
        now = timezone.now()
        run = (
            CollectionSchemaGenerationRun.objects.select_for_update()
            .filter(
                id=run_id,
                status="running",
                lease_token=lease_token,
                lease_expires_at__gt=now,
            )
            .first()
        )
        if run is None:
            return False
        run.status = "queued"
        run.statistics = {
            **run.statistics,
            "inference_retries": run.statistics.get("inference_retries", 0) + 1,
        }
        run.lease_token = None
        run.lease_expires_at = None
        run.save(
            update_fields=[
                "status",
                "statistics",
                "lease_token",
                "lease_expires_at",
            ]
        )
        return True








@shared_task(
    bind=True,
    name="apps.collections.tasks.schema_generation.generate_collection_schema_task",
    queue=_QUEUE,
    serializer="json",
    max_retries=_MAX_RETRIES,
    acks_late=True,
    reject_on_worker_lost=True,
    ignore_result=True,
)
def generate_collection_schema_task(self, run_id: str) -> None:
    """Generate one draft, with source and draft-revision fences at each boundary."""

    try:
        parsed_run_id = _canonical_run_id(run_id)
    except ValueError as exc:
        _safe_log_failure("local_inference_failed", exc)
        raise
    try:
        claim = _claim_run(parsed_run_id)
    except _LeaseBusy as exc:
        # A worker-loss redelivery is the only remaining message in some cases.
        # Keep it deliverable even when inference retries have been exhausted.
        raise self.retry(
            countdown=exc.retry_after,
            max_retries=max(self.max_retries, int(self.request.retries) + 1),
            queue=_QUEUE,
        )
    if claim is None:
        return None
    run, lease_token = claim.run, claim.lease_token
    if not _generation_enabled():
        _fail_or_retry(self, parsed_run_id, lease_token, "disabled", _LeaseLost())
        return None
    try:
        source_signature = _prepare_run_source(run, lease_token)
        config = load_schema_generation_config()
        samples = sample_collection_chunks(
            run.collection_id, config.max_chunks, config.max_characters
        )
        if not samples:
            _fail_or_retry(
                self, parsed_run_id, lease_token, "no_collection_text", _LeaseLost()
            )
            return None
        candidate = generate_schema_candidate(samples)
        definitions, statistics = collect_candidate_evidence(candidate, samples)
        _write_draft_with_source_fence(
            parsed_run_id,
            run.collection_id,
            source_signature,
            lease_token,
            definitions,
            statistics,
        )
    except _SourceChanged:
        _defer_source(self, parsed_run_id, lease_token)
    except _LeaseLost as exc:
        _retry_after_lease_loss(
            self, parsed_run_id, lease_token, "local_inference_failed", exc
        )
    except InvalidSchemaCandidate as exc:
        if _fail_or_retry(self, parsed_run_id, lease_token, "invalid_candidate", exc):
            _safe_log_failure("invalid_candidate", exc)
    except Exception as exc:
        from apps.collections.services.schema import SchemaGenerationDraftConflict

        if isinstance(exc, SchemaGenerationDraftConflict):
            if _fail_or_retry(self, parsed_run_id, lease_token, "draft_conflict", exc):
                _safe_log_failure("draft_conflict", exc)
            return None
        _retry_or_fail(
            self,
            parsed_run_id,
            lease_token,
            exc,
            inference_retries=run.statistics.get("inference_retries", 0),
        )
    return None


__all__ = ["enqueue_schema_generation", "generate_collection_schema_task"]

from .schema_generation_lease import (
    _claim_run as _claim_run,
    _defer_source as _defer_source,
    _prepare_run_source as _prepare_run_source,
    _write_draft_with_source_fence as _write_draft_with_source_fence,
)

from .schema_generation_retries import (
    _fail_or_retry as _fail_or_retry,
    _retry_after_lease_loss as _retry_after_lease_loss,
    _retry_or_fail as _retry_or_fail,
)
