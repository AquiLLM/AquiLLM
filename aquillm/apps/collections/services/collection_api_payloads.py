"""Pure helpers for collection API JSON payloads (types, ingestion metadata, figure trees)."""
from __future__ import annotations

from apps.collections.models import Collection
from apps.documents.models import DocumentFigure
from apps.ingestion.models import IngestionBatchItem


def normalized_type_label(normalized_type: str) -> str:
    raw = (normalized_type or "").strip().lower()
    if not raw:
        return ""
    if raw == "document_figure":
        return "DocumentFigure"
    return raw.upper()


def raw_text_type_overrides(collection: Collection) -> dict[str, str]:
    """Map document UUID -> parser-derived display type for RawTextDocument rows."""
    overrides: dict[str, str] = {}
    items = IngestionBatchItem.objects.filter(
        batch__collection=collection,
        status=IngestionBatchItem.Status.SUCCESS,
    ).only("parser_metadata")

    for item in items:
        parser_metadata = item.parser_metadata or {}
        if not isinstance(parser_metadata, dict):
            continue
        outputs = parser_metadata.get("outputs") or []
        if not isinstance(outputs, list):
            continue

        for output in outputs:
            if not isinstance(output, dict):
                continue
            document_id = str(output.get("document_id") or "").strip()
            normalized_type = str(output.get("normalized_type") or "").strip()
            if not document_id or not normalized_type:
                continue
            overrides[document_id] = normalized_type_label(normalized_type)

    return overrides


def _norm_title_key(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _source_title_from_figure_title(title: str) -> str:
    marker = " - Figure "
    if marker not in (title or ""):
        return ""
    return title.split(marker, 1)[0].strip()


def child_collection_parent_document_ids(
    child_collection_ids: list[int],
    valid_document_ids: set[str],
    document_title_to_id: dict[str, str],
) -> dict[int, str]:
    """Map child collection id -> parent document id (figures linking back to a document)."""
    if not child_collection_ids:
        return {}

    mapping: dict[int, str] = {}
    rows = (
        DocumentFigure.objects.filter(
            collection_id__in=child_collection_ids,
        )
        .values("collection_id", "parent_object_id", "title")
    )
    for row in rows:
        collection_id = int(row["collection_id"])
        if collection_id in mapping:
            continue
        parent_document_id = str(row["parent_object_id"] or "").strip()
        if parent_document_id and (not valid_document_ids or parent_document_id in valid_document_ids):
            mapping[collection_id] = parent_document_id
            continue

        source_title = _source_title_from_figure_title(str(row.get("title") or ""))
        if not source_title:
            continue
        inferred_id = document_title_to_id.get(_norm_title_key(source_title))
        if inferred_id:
            mapping[collection_id] = inferred_id

    return mapping


__all__ = [
    "child_collection_parent_document_ids",
    "normalized_type_label",
    "raw_text_type_overrides",
]
