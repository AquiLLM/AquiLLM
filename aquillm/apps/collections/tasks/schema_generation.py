"""Durable, local-only Celery boundary for collection schema generation."""
from __future__ import annotations

import os
import uuid

import structlog
from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from apps.collections.services.schema_generation import (
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
_QUEUE = settings.KG_EXTRACTION_QUEUE


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
            .select_related("collection")
            .filter(id=run_id)
            .first()
        )
        if run is None or run.status != "queued":
            return None
        run.status = "running"
        run.started_at = timezone.now()
        run.error_code = ""
        run.save(update_fields=["status", "started_at", "error_code"])
        return run


def _fail_run(run_id: uuid.UUID, error_code: str) -> None:
    from apps.collections.models import CollectionSchemaGenerationRun

    CollectionSchemaGenerationRun.objects.filter(
        id=run_id, status__in=("queued", "running")
    ).update(status="failed", error_code=error_code, completed_at=timezone.now())


def _safe_log_failure(error_code: str, exc: BaseException | None = None) -> None:
    """Log only a public code and exception class; never samples, prompts, or secrets."""

    logger.error(
        "obs.collections.schema_generation_failed",
        error_code=error_code,
        error_type=None if exc is None else type(exc).__name__,
    )


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
    run = _claim_run(parsed_run_id)
    if run is None:
        return None
    if not _generation_enabled():
        _fail_run(parsed_run_id, "disabled")
        return None
    try:
        if collection_source_signature(run.collection_id) != run.source_signature:
            _fail_run(parsed_run_id, "source_changed")
            return None
        config = load_schema_generation_config()
        samples = sample_collection_chunks(
            run.collection_id, config.max_chunks, config.max_characters
        )
        if not samples:
            _fail_run(parsed_run_id, "no_collection_text")
            return None
        candidate = generate_schema_candidate(samples)
        definitions, statistics = collect_candidate_evidence(candidate, samples)
        # Task 1 owns persistent canonicalization and draft conflict semantics.
        from apps.collections.services.schema import canonicalize_definitions, write_generated_draft

        if collection_source_signature(run.collection_id) != run.source_signature:
            _fail_run(parsed_run_id, "source_changed")
            return None
        write_generated_draft(
            parsed_run_id,
            canonicalize_definitions(definitions),
            statistics,
        )
    except InvalidSchemaCandidate as exc:
        _fail_run(parsed_run_id, "invalid_candidate")
        _safe_log_failure("invalid_candidate", exc)
    except Exception as exc:
        from apps.collections.services.schema import SchemaGenerationDraftConflict

        if isinstance(exc, SchemaGenerationDraftConflict):
            _fail_run(parsed_run_id, "draft_conflict")
            _safe_log_failure("draft_conflict", exc)
            return None
        retry_count = int(getattr(self.request, "retries", 0))
        if retry_count < self.max_retries:
            raise self.retry(exc=exc, countdown=_RETRY_COUNTDOWN_SECONDS)
        _fail_run(parsed_run_id, "local_inference_failed")
        _safe_log_failure("local_inference_failed", exc)
    return None


__all__ = ["enqueue_schema_generation", "generate_collection_schema_task"]
