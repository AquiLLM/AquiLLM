"""Retry or finalize a fenced collection schema generation delivery."""

from __future__ import annotations

import uuid

from . import schema_generation as core
from .schema_generation import _QUEUE, _RETRY_COUNTDOWN_SECONDS


def _retry_after_lease_loss(
    task, run_id: uuid.UUID, lease_token: uuid.UUID, error_code: str, exc: BaseException
) -> bool:
    """Redeliver a stale owner, or fail only its own expired lease at exhaustion."""

    if int(getattr(task.request, "retries", 0)) < task.max_retries:
        raise task.retry(exc=exc, countdown=_RETRY_COUNTDOWN_SECONDS, queue=_QUEUE)
    return core._fail_expired_run(run_id, error_code, lease_token)


def _fail_or_retry(
    task, run_id: uuid.UUID, lease_token: uuid.UUID, error_code: str, exc: BaseException
) -> bool:
    """Record a terminal result only for this owner, otherwise preserve delivery."""

    if core._fail_run(run_id, error_code, lease_token):
        return True
    return core._retry_after_lease_loss(task, run_id, lease_token, error_code, exc)


def _retry_or_fail(
    task,
    run_id: uuid.UUID,
    lease_token: uuid.UUID,
    exc: BaseException,
    *,
    inference_retries: int | None = None,
) -> bool:
    """Release an owned retry attempt, or preserve a stale delivery for recovery."""

    retry_count = (
        int(getattr(task.request, "retries", 0))
        if inference_retries is None
        else inference_retries
    )
    if retry_count < task.max_retries:
        if core._release_lease_for_retry(run_id, lease_token):
            raise task.retry(
                exc=exc,
                countdown=_RETRY_COUNTDOWN_SECONDS,
                max_retries=max(task.max_retries, int(task.request.retries) + 1),
                queue=_QUEUE,
            )
        return core._retry_after_lease_loss(
            task, run_id, lease_token, "local_inference_failed", exc
        )
    if core._fail_run(run_id, "local_inference_failed", lease_token):
        core._safe_log_failure("local_inference_failed", exc)
        return True
    if core._fail_expired_run(run_id, "local_inference_failed", lease_token):
        core._safe_log_failure("local_inference_failed", exc)
        return True
    return False
