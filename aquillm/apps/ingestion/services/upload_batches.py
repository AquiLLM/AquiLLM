"""Queue uploaded files into an ingestion batch."""
from __future__ import annotations

from typing import Any

from apps.ingestion.models import IngestionBatch, IngestionBatchItem
from aquillm.tasks import ingest_uploaded_file_task


def enqueue_upload_batch_files(
    user: Any,
    collection: Any,
    files: list[Any],
    *,
    max_files: int,
    max_file_bytes: int,
) -> tuple[dict, int]:
    """
    Create batch items and dispatch Celery tasks.

    Returns ``(response_body, http_status)`` for ``JsonResponse``.
    """
    if len(files) > max_files:
        return ({"error": f"Too many files. Maximum is {max_files} per batch."}, 400)

    batch = IngestionBatch.objects.create(user=user, collection=collection)
    queued_items: list[dict[str, object]] = []
    rejected_items: list[dict[str, object]] = []

    for upload in files:
        size = int(getattr(upload, "size", 0) or 0)
        if size <= 0:
            rejected_items.append({"filename": upload.name, "error": "Empty file."})
            continue
        if size > max_file_bytes:
            rejected_items.append(
                {
                    "filename": upload.name,
                    "error": f"File exceeds INGEST_MAX_FILE_BYTES ({max_file_bytes}).",
                }
            )
            continue

        item = IngestionBatchItem.objects.create(
            batch=batch,
            source_file=upload,
            original_filename=upload.name,
            content_type=getattr(upload, "content_type", "") or "",
            file_size_bytes=size,
            status=IngestionBatchItem.Status.QUEUED,
        )
        ingest_uploaded_file_task.delay(item.id)
        queued_items.append({"id": item.id, "filename": item.original_filename, "status": item.status})

    if not queued_items:
        batch.delete()
        return (
            {
                "status": "error",
                "queued_count": 0,
                "rejected_count": len(rejected_items),
                "items": [],
                "rejected": rejected_items,
            },
            400,
        )

    return (
        {
            "batch_id": batch.id,
            "status": "queued",
            "queued_count": len(queued_items),
            "rejected_count": len(rejected_items),
            "items": queued_items,
            "rejected": rejected_items,
        },
        202,
    )


__all__ = ["enqueue_upload_batch_files"]
