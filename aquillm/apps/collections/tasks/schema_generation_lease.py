"""Claim collection schema runs and fence source/draft writes."""

from __future__ import annotations

import uuid
from math import ceil

from django.db import transaction
from django.utils import timezone

from . import schema_generation as core
from .schema_generation import (
    _LEASE_DURATION,
    _MAX_SOURCE_DEFERRALS,
    _QUEUE,
    _RETRY_COUNTDOWN_SECONDS,
    _SOURCE_SETTLING_DURATION,
    _LeaseBusy,
    _LeaseLost,
    _RunClaim,
    _SourceChanged,
)


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
        if (
            run.status == "running"
            and run.lease_expires_at is not None
            and run.lease_expires_at > now
        ):
            raise _LeaseBusy(max(1, ceil((run.lease_expires_at - now).total_seconds())))
        lease_token = uuid.uuid4()
        if run.status == "queued":
            run.status = "running"
            if run.started_at is None:
                run.started_at = now
        run.error_code = ""
        run.lease_token = lease_token
        run.lease_expires_at = now + _LEASE_DURATION
        run.save(
            update_fields=[
                "status",
                "started_at",
                "error_code",
                "lease_token",
                "lease_expires_at",
            ]
        )
        return _RunClaim(run=run, lease_token=lease_token)


def _prepare_run_source(run, lease_token: uuid.UUID) -> str:
    """Refresh only the source snapshot, never the user's requested draft fence."""

    from apps.collections.models import (
        Collection,
        CollectionSchemaDraft,
        CollectionSchemaGenerationRun,
    )
    from apps.collections.services.schema import SchemaGenerationDraftConflict

    with transaction.atomic():
        Collection.objects.select_for_update().get(pk=run.collection_id)
        signature = core._locked_collection_source_signature(run.collection_id)
        current = (
            CollectionSchemaGenerationRun.objects.select_for_update()
            .filter(
                pk=run.pk,
                collection_id=run.collection_id,
                status="running",
                lease_token=lease_token,
                lease_expires_at__gt=timezone.now(),
            )
            .first()
        )
        if current is None:
            raise _LeaseLost()
        draft = (
            CollectionSchemaDraft.objects.select_for_update()
            .filter(collection_id=run.collection_id)
            .first()
        )
        if (draft.pk if draft else None) != current.base_draft_id or (
            draft.revision if draft else None
        ) != current.base_draft_revision:
            raise SchemaGenerationDraftConflict("draft_conflict")
        if core.collection_ingestion_pending(run.collection_id):
            raise _SourceChanged()
        current.source_signature = signature
        current.save(update_fields=["source_signature"])
        return signature


def _defer_source(task, run_id: uuid.UUID, lease_token: uuid.UUID) -> None:
    """Release the worker while uploads settle, with durable time/attempt bounds."""

    from apps.collections.models import CollectionSchemaGenerationRun

    with transaction.atomic():
        now = timezone.now()
        run = (
            CollectionSchemaGenerationRun.objects.select_for_update()
            .filter(
                pk=run_id,
                status="running",
                lease_token=lease_token,
                lease_expires_at__gt=now,
            )
            .first()
        )
        if run is not None:
            deferrals = run.statistics.get("source_deferrals", 0)
            if (
                deferrals >= _MAX_SOURCE_DEFERRALS
                or now - run.started_at >= _SOURCE_SETTLING_DURATION
            ):
                run.status = "failed"
                run.error_code = "source_changed"
                run.completed_at = now
            else:
                run.status = "queued"
                run.statistics = {**run.statistics, "source_deferrals": deferrals + 1}
            run.lease_token = None
            run.lease_expires_at = None
            run.save(
                update_fields=[
                    "status",
                    "error_code",
                    "completed_at",
                    "statistics",
                    "lease_token",
                    "lease_expires_at",
                ]
            )
    if run is None:
        core._retry_after_lease_loss(
            task, run_id, lease_token, "source_changed", _SourceChanged()
        )
    elif run.status == "queued":
        raise task.retry(
            countdown=_RETRY_COUNTDOWN_SECONDS,
            max_retries=max(task.max_retries, int(task.request.retries) + 1),
            queue=_QUEUE,
        )


def _write_draft_with_source_fence(
    run_id: uuid.UUID,
    collection_id: int,
    expected_signature: str,
    lease_token: uuid.UUID,
    definitions: dict,
    statistics: dict,
):
    """Fence source writes with parent/source locks and the current execution lease."""

    from apps.collections.models import Collection, CollectionSchemaGenerationRun
    from apps.collections.services.schema import (
        canonicalize_definitions,
        write_generated_draft,
    )

    with transaction.atomic():
        Collection.objects.select_for_update().get(pk=collection_id)
        if (
            core._locked_collection_source_signature(collection_id)
            != expected_signature
        ):
            raise _SourceChanged()
        if core.collection_ingestion_pending(collection_id):
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
        draft = write_generated_draft(
            run_id, canonicalize_definitions(definitions), statistics
        )
        run.lease_token = None
        run.lease_expires_at = None
        run.save(update_fields=["lease_token", "lease_expires_at"])
        return draft
