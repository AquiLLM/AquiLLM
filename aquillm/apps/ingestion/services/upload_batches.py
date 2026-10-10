"""Queue uploaded files into an ingestion batch."""
from __future__ import annotations

from typing import Any
from django.db import transaction
from django.core.files.storage import default_storage

from apps.ingestion.models import IngestionBatch, IngestionBatchItem
from aquillm.tasks import ingest_uploaded_file_task
from .output_dispatch import dispatch_item


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

    queued_items: list[dict[str, object]] = []
    rejected_items: list[dict[str, object]] = []
    prepared = []

    for source_index, upload in enumerate(files):
        size = int(getattr(upload, "size", 0) or 0)
        if size <= 0:
            rejected_items.append({"filename": upload.name, "source_index": source_index, "error": "Empty file."})
            continue
        if size > max_file_bytes:
            rejected_items.append(
                {
                    "filename": upload.name,
                    "source_index": source_index,
                    "error": f"File exceeds INGEST_MAX_FILE_BYTES ({max_file_bytes}).",
                }
            )
            continue

        field = IngestionBatchItem._meta.get_field('source_file')
        stored = default_storage.save(field.generate_filename(None, upload.name), upload, max_length=500)
        prepared.append((source_index, upload.name, getattr(upload, 'content_type', '') or '', size, stored))

    if prepared:
        with transaction.atomic():
            batch = IngestionBatch.objects.create(user=user, collection=collection)
            for source_index, name, content_type, size, stored in prepared:
                item = IngestionBatchItem.objects.create(batch=batch, source_file=stored,
                    original_filename=name, content_type=content_type, file_size_bytes=size,
                    status=IngestionBatchItem.Status.QUEUED)
                transaction.on_commit(lambda pk=item.id: dispatch_item(pk), robust=True)
                queued_items.append({'id': item.id, 'filename': name, 'source_index': source_index, 'status': item.status})

    if not queued_items:
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
