"""Stream collection source identities and detect unfinished ingestion."""
from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterator

from . import schema_generation as core


def _collection_source_documents(
    collection_id: int, *, for_update: bool = False, include_incomplete: bool = False
) -> Iterator[dict[str, object]]:
    """Stream text-free document identities, locking every source row when requested."""

    from django.apps import apps

    iterators = []
    for name in (
        "PDFDocument",
        "TeXDocument",
        "RawTextDocument",
        "VTTDocument",
        "HandwrittenNotesDocument",
        "ImageUploadDocument",
        "MediaUploadDocument",
        "DocumentFigure",
    ):
        model = apps.get_model("apps_documents", name)
        filters = {"collection_id": collection_id}
        if not include_incomplete:
            filters["ingestion_complete"] = True
        queryset = model.objects.filter(**filters).order_by("id")
        if for_update:
            queryset = queryset.select_for_update()
        fields = (
            ("id", "full_text_hash", "ingestion_complete")
            if include_incomplete
            else ("id", "full_text_hash")
        )
        iterators.append(queryset.values(*fields).iterator())
    yield from heapq.merge(*iterators, key=lambda record: str(record["id"]))


def _completed_collection_documents(
    collection_id: int, *, for_update: bool = False
) -> Iterator[dict[str, object]]:
    """Stream completed document identities; never read document text here."""

    yield from core._collection_source_documents(collection_id, for_update=for_update)


def collection_source_signature(collection_id) -> str:
    """Return a text-free signature for completed collection document content."""

    if type(collection_id) is not int or collection_id <= 0:
        raise ValueError("collection_id must be a positive database integer")
    digest = hashlib.sha256()
    for document in core._completed_collection_documents(collection_id):
        digest.update(str(document["id"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(document["full_text_hash"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def collection_ingestion_pending(collection_id: int) -> bool:
    """Include queued uploads that have not created document rows yet."""

    from apps.ingestion.models import IngestionBatchItem

    if IngestionBatchItem.objects.filter(
        batch__collection_id=collection_id,
        status__in=(
            IngestionBatchItem.Status.QUEUED,
            IngestionBatchItem.Status.PROCESSING,
        ),
    ).exists():
        return True
    return any(
        not document["ingestion_complete"]
        for document in core._collection_source_documents(
            collection_id, include_incomplete=True
        )
    )


def _locked_collection_source_signature(collection_id: int) -> str:
    """Lock all source rows, then hash only completed document identities."""

    digest = hashlib.sha256()
    for document in core._collection_source_documents(
        collection_id, for_update=True, include_incomplete=True
    ):
        if not document["ingestion_complete"]:
            continue
        digest.update(str(document["id"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(document["full_text_hash"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
