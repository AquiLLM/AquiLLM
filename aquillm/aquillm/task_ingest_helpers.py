"""Helpers for ingestion batch Celery task (figures, media refs)."""
from __future__ import annotations

import hashlib


def media_uniqueness_ref(item_id: int, payload_index: int, modality: str, media_bytes: bytes | None) -> str:
    digest = hashlib.sha256(media_bytes or b"").hexdigest()[:12]
    return f"[ref:{item_id}:{payload_index}:{modality}:{digest}]"


def fallback_media_text(item_id: int, payload_index: int, modality: str, media_bytes: bytes | None) -> str:
    return (
        "No text could be extracted from this media. "
        + media_uniqueness_ref(
            item_id=item_id,
            payload_index=payload_index,
            modality=modality,
            media_bytes=media_bytes,
        )
    )


def normalize_source_key(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def infer_source_title_from_figure_title(title: str) -> str:
    marker = " - Figure "
    if marker not in (title or ""):
        return ""
    return title.split(marker, 1)[0].strip()


def figures_subcollection_name(source_doc_title: str, max_len: int = 100) -> str:
    suffix = " - Figures"
    base = (source_doc_title or "").strip() or "Document"
    allowed = max_len - len(suffix)
    if allowed <= 0:
        return "Figures"[:max_len]

    if len(base) > allowed:
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:6]
        trunc = max(1, allowed - 7)
        base = f"{base[:trunc].rstrip()}~{digest}"
    return f"{base}{suffix}"


def ensure_child_collection_permissions(CollectionPermission, parent_collection, child_collection) -> None:
    permissions = CollectionPermission.objects.filter(collection=parent_collection).values("user_id", "permission")
    for permission in permissions:
        CollectionPermission.objects.update_or_create(
            user_id=permission["user_id"],
            collection=child_collection,
            defaults={"permission": permission["permission"]},
        )


def get_or_create_figures_subcollection(Collection, CollectionPermission, parent_collection, source_doc_title: str):
    from django.db import IntegrityError

    child_name = figures_subcollection_name(source_doc_title)
    try:
        child_collection, _ = Collection.objects.get_or_create(
            name=child_name,
            parent=parent_collection,
        )
    except IntegrityError:
        child_collection = Collection.objects.get(name=child_name, parent=parent_collection)

    ensure_child_collection_permissions(
        CollectionPermission=CollectionPermission,
        parent_collection=parent_collection,
        child_collection=child_collection,
    )
    return child_collection


__all__ = [
    "ensure_child_collection_permissions",
    "fallback_media_text",
    "figures_subcollection_name",
    "get_or_create_figures_subcollection",
    "infer_source_title_from_figure_title",
    "media_uniqueness_ref",
    "normalize_source_key",
]
