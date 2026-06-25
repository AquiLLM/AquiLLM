"""Helpers for ingestion batch Celery task (figures, media refs)."""
from __future__ import annotations

import hashlib


def sanitize_db_text(value) -> str:
    """
    PostgreSQL text/varchar cannot contain NUL (0x00) characters.

    Some extractors (notably PDF text) can yield NULs; strip them so saves don't
    hard-fail with: "A string literal cannot contain NUL (0x00) characters."
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return text.replace("\x00", "")


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
    """Make child collection permissions exactly match the parent (add, update, remove)."""
    permissions = list(
        CollectionPermission.objects.filter(collection=parent_collection).values("user_id", "permission")
    )
    parent_user_ids = {p["user_id"] for p in permissions}
    child_qs = CollectionPermission.objects.filter(collection=child_collection)
    if parent_user_ids:
        child_qs.exclude(user_id__in=parent_user_ids).delete()
    else:
        child_qs.delete()
    for permission in permissions:
        CollectionPermission.objects.update_or_create(
            user_id=permission["user_id"],
            collection=child_collection,
            defaults={"permission": permission["permission"]},
        )


def is_figures_derived_subcollection_name(name: str) -> bool:
    """True for collections created by get_or_create_figures_subcollection (document-derived figures)."""
    n = (name or "").strip()
    return n.endswith(" - Figures") or n == "Figures"


def sync_figure_subcollection_permissions_from_parent(parent_collection) -> None:
    """Copy parent sharing to direct children that hold extracted document figures."""
    from apps.collections.models import CollectionPermission

    for child in parent_collection.children.all():
        if not is_figures_derived_subcollection_name(child.name):
            continue
        ensure_child_collection_permissions(
            CollectionPermission,
            parent_collection=parent_collection,
            child_collection=child,
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
    "is_figures_derived_subcollection_name",
    "media_uniqueness_ref",
    "normalize_source_key",
    "sanitize_db_text",
    "sync_figure_subcollection_permissions_from_parent",
]
