"""Durable, local-only Celery boundary for collection schema generation."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import timedelta

import structlog
from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from apps.collections.services.schema_generation import (
    _locked_collection_source_signature,
    InvalidSchemaCandidate,
    collection_source_signature,
    collect_candidate_evidence,
    generate_schema_candidate,
    load_schema_generation_config,
    sample_collection_chunks,
)

logger = structlog.stdlib.get_logger(__name__)
_MAX_RETRIES = 3
_RETRY_COUNTDOWN_SECONDS = 30
_LEASE_DURATION = timedelta(minutes=10)
_QUEUE = settings.KG_EXTRACTION_QUEUE


class _SourceChanged(RuntimeError):
    """The final source-document lock fence rejected a stale generation run."""


class _LeaseLost(RuntimeError):
    """A newer delivery owns the run, so this stale delivery must do nothing."""


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
    """Publish an exact durable run identifier to the isolated graph queue."""

    parsed = run_id if isinstance(run_id, uuid.UUID) else _canonical_run_id(run_id)
    generate_collection_schema_task.delay(str(parsed))


def _claim_run(run_id: uuid.UUID):
    from apps.collections.models import CollectionSchemaGenerationRun

    with transaction.atomic():
        run = (
            CollectionSchemaGenerationRun.objects.select_for_update()
            .filter(id=run_id)
            .first()
        )
        if run is None or run.status not in {"queued", "running"}:
            return None
        now = timezone.now()
        if run.status == "running" and run.lease_expires_at is not None and run.lease_expires_at > now:
            return None
        lease_token = uuid.uuid4()
        if run.status == "queued":
            run.status = "running"
            run.started_at = now
        run.error_code = ""
        run.lease_token = lease_token
        run.lease_expires_at = now + _LEASE_DURATION
        run.save(update_fields=["status", "started_at", "error_code", "lease_token", "lease_expires_at"])
        return _RunClaim(run=run, lease_token=lease_token)


def _fail_run(run_id: uuid.UUID, error_code: str, lease_token: uuid.UUID) -> None:
    from apps.collections.models import CollectionSchemaGenerationRun

    filters = {
        "id": run_id,
        "status__in": ("queued", "running"),
        "lease_token": lease_token,
        "lease_expires_at__gt": timezone.now(),
    }
    CollectionSchemaGenerationRun.objects.filter(**filters).update(
        status="failed", error_code=error_code, completed_at=timezone.now(),
        lease_token=None, lease_expires_at=None,
    )


def _safe_log_failure(error_code: str, exc: BaseException | None = None) -> None:
    """Log only a public code and exception class; never samples, prompts, or secrets."""

    logger.error(
        "obs.collections.schema_generation_failed",
        error_code=error_code,
        error_type=None if exc is None else type(exc).__name__,
    )


def _write_draft_with_source_fence(
    run_id: uuid.UUID, collection_id: int, expected_signature: str, lease_token: uuid.UUID,
    definitions: dict, statistics: dict,
):
    """Fence source writes with parent/source locks and the current execution lease."""

    from apps.collections.models import Collection, CollectionSchemaGenerationRun
    from apps.collections.services.schema import canonicalize_definitions, write_generated_draft

    with transaction.atomic():
        Collection.objects.select_for_update().get(pk=collection_id)
        if _locked_collection_source_signature(collection_id) != expected_signature:
            raise _SourceChanged()
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
            raise _LeaseLost()
        run.lease_expires_at = now + _LEASE_DURATION
        run.save(update_fields=["lease_expires_at"])
        draft = write_generated_draft(run_id, canonicalize_definitions(definitions), statistics)
        run.lease_token = None
        run.lease_expires_at = None
        run.save(update_fields=["lease_token", "lease_expires_at"])
        return draft


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
        run.lease_token = None
        run.lease_expires_at = None
        run.save(update_fields=["status", "lease_token", "lease_expires_at"])
        return True


def _retry_or_fail(task, run_id: uuid.UUID, lease_token: uuid.UUID, exc: BaseException) -> None:
    """Leave a live run resumable for retry; record only exhausted local failures."""

    retry_count = int(getattr(task.request, "retries", 0))
    if retry_count < task.max_retries:
        if _release_lease_for_retry(run_id, lease_token):
            raise task.retry(exc=exc, countdown=_RETRY_COUNTDOWN_SECONDS)
        return
    _fail_run(run_id, "local_inference_failed", lease_token)
    _safe_log_failure("local_inference_failed", exc)


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
    claim = _claim_run(parsed_run_id)
    if claim is None:
        return None
    run, lease_token = claim.run, claim.lease_token
    if not _generation_enabled():
        _fail_run(parsed_run_id, "disabled", lease_token)
        return None
    try:
        if collection_source_signature(run.collection_id) != run.source_signature:
            _fail_run(parsed_run_id, "source_changed", lease_token)
            return None
        config = load_schema_generation_config()
        samples = sample_collection_chunks(
            run.collection_id, config.max_chunks, config.max_characters
        )
        if not samples:
            _fail_run(parsed_run_id, "no_collection_text", lease_token)
            return None
        candidate = generate_schema_candidate(samples)
        definitions, statistics = collect_candidate_evidence(candidate, samples)
        _write_draft_with_source_fence(
            parsed_run_id, run.collection_id, run.source_signature, lease_token, definitions, statistics
        )
    except _SourceChanged:
        _fail_run(parsed_run_id, "source_changed", lease_token)
    except _LeaseLost:
        return None
    except InvalidSchemaCandidate as exc:
        _fail_run(parsed_run_id, "invalid_candidate", lease_token)
        _safe_log_failure("invalid_candidate", exc)
    except Exception as exc:
        from apps.collections.services.schema import SchemaGenerationDraftConflict

        if isinstance(exc, SchemaGenerationDraftConflict):
            _fail_run(parsed_run_id, "draft_conflict", lease_token)
            _safe_log_failure("draft_conflict", exc)
            return None
        _retry_or_fail(self, parsed_run_id, lease_token, exc)
    return None


__all__ = ["enqueue_schema_generation", "generate_collection_schema_task"]
