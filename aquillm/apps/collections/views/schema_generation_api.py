"""Collection schema generation HTTP lifecycle endpoints."""

from __future__ import annotations

import structlog
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from apps.collections.models import (
    CollectionSchemaGenerationRun,
)
from apps.collections.services import schema as schema_service

from . import schema_api as core
from .schema_api_helpers import (
    error_response,
    load_body,
)

logger = structlog.stdlib.get_logger(__name__)


def _enqueue_generation_safely(run_id: str) -> None:
    """Keep broker failures bounded and leave the collection immediately retryable."""

    try:
        core.enqueue_schema_generation(run_id)
    except Exception as exc:  # Celery transports expose several broker exceptions.
        core.CollectionSchemaGenerationRun.objects.filter(
            pk=run_id,
            status=core.CollectionSchemaGenerationRun.Status.QUEUED,
        ).update(
            status=core.CollectionSchemaGenerationRun.Status.FAILED,
            error_code="local_inference_failed",
            completed_at=timezone.now(),
        )
        logger.error(
            "obs.collections.schema_generation_enqueue_failed",
            error_type=type(exc).__name__,
        )


@login_required
@require_http_methods(["POST"])
def schema_generate(request, col_id: int):
    collection = core._collection(col_id)
    if denied := core.require_edit(collection, request.user):
        return denied
    try:
        if load_body(request):
            raise schema_service.SchemaOperationError("invalid_body")
        with transaction.atomic():
            locked_collection = core.Collection.objects.select_for_update().get(
                pk=collection.pk
            )
            draft = (
                core.CollectionSchemaDraft.objects.select_for_update()
                .filter(collection=locked_collection)
                .first()
            )
            draft_definitions = (
                schema_service.canonicalize_definitions(draft.definitions)
                if draft is not None
                else {"entities": [], "relations": []}
            )
            if draft is not None and any(draft_definitions.values()):
                raise schema_service.SchemaOperationError("draft_exists", status=409)
            base_draft_id = draft.pk if draft is not None else None
            base_draft_revision = draft.revision if draft is not None else None
            source_signature = core._locked_collection_source_signature(
                locked_collection.pk
            )
            run = (
                core.CollectionSchemaGenerationRun.objects.select_for_update()
                .filter(
                    collection=locked_collection,
                    status__in=(
                        core.CollectionSchemaGenerationRun.Status.QUEUED,
                        core.CollectionSchemaGenerationRun.Status.RUNNING,
                    ),
                )
                .first()
            )
            if (
                run is not None
                and run.status == "running"
                and (
                    run.lease_expires_at is None
                    or run.lease_expires_at <= timezone.now()
                )
            ):
                # Revoke the old owner under its row lock before queuing a new run.
                # Its late output and redeliveries can no longer claim or write.
                run.status = "failed"
                run.error_code = "local_inference_failed"
                run.completed_at = timezone.now()
                run.lease_token = None
                run.lease_expires_at = None
                run.save(
                    update_fields=(
                        "status",
                        "error_code",
                        "completed_at",
                        "lease_token",
                        "lease_expires_at",
                        "updated_at",
                    )
                )
                run = None
            if run is not None and run.source_signature != source_signature:
                raise schema_service.SchemaOperationError("source_changed", status=409)
            legacy_run_rebound = False
            if (
                run is not None
                and draft is not None
                and run.base_draft_id is None
                and run.base_draft_revision is None
            ):
                run.base_draft_id = base_draft_id
                run.base_draft_revision = base_draft_revision
                run.save(
                    update_fields=("base_draft_id", "base_draft_revision", "updated_at")
                )
                legacy_run_rebound = True
            if run is not None and (
                run.base_draft_id != base_draft_id
                or run.base_draft_revision != base_draft_revision
            ):
                raise schema_service.SchemaOperationError("draft_exists", status=409)
            if run is None:
                run = core.CollectionSchemaGenerationRun.objects.create(
                    collection=locked_collection,
                    requested_by=request.user,
                    source_signature=source_signature,
                    base_draft_id=base_draft_id,
                    base_draft_revision=base_draft_revision,
                )
                run_id = str(run.pk)
                transaction.on_commit(
                    lambda run_id=run_id: core._enqueue_generation_safely(run_id)
                )
            elif legacy_run_rebound:
                run_id = str(run.pk)
                transaction.on_commit(
                    lambda run_id=run_id: core._enqueue_generation_safely(run_id)
                )
    except schema_service.SchemaOperationError as exc:
        return error_response(exc)
    return JsonResponse(
        {
            "run_id": str(run.pk),
            "status": run.status,
            "status_url": reverse(
                "api_collection_schema_generation_status",
                kwargs={"col_id": collection.pk, "run_id": run.pk},
            ),
        },
        status=202,
    )


@login_required
@require_http_methods(["GET"])
def schema_generation_status(request, col_id: int, run_id):
    collection = core._collection(col_id)
    if denied := core.require_view(collection, request.user):
        return denied
    run = get_object_or_404(
        CollectionSchemaGenerationRun,
        pk=run_id,
        collection=collection,
    )
    payload = {
        "run_id": str(run.pk),
        "status": run.status,
        "error_code": run.error_code or None,
        "statistics": run.statistics,
    }
    if run.status == core.CollectionSchemaGenerationRun.Status.SUCCEEDED:
        payload["workspace"] = core.workspace_envelope(collection, request.user)
    return JsonResponse(payload)
