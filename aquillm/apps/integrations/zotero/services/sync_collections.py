"""Sync Zotero collections into AquiLLM with parent/child hierarchy."""
from __future__ import annotations

import structlog

from django.contrib.auth.models import User

from apps.collections.models import Collection, CollectionPermission

from aquillm.zotero_client import ZoteroAPIClient

logger = structlog.stdlib.get_logger(__name__)


def sync_collections_with_hierarchy(
    client: ZoteroAPIClient,
    user: User,
    collection_map: dict,
    library_id=None,
    library_type: str = "user",
):
    """
    Sync collections while preserving parent/child hierarchy.

    Collections may be deleted at end of sync if empty (see main task).

    Returns:
        Tuple of (collections_created, collections_updated, errors)
    """
    collections_created = 0
    collections_updated = 0
    errors = 0

    zotero_collections = client.get_collections(group_id=library_id)

    collections_with_parents = []
    collections_without_parents = []

    for zotero_col in zotero_collections:
        col_data = zotero_col["data"]
        if col_data.get("parentCollection"):
            collections_with_parents.append(zotero_col)
        else:
            collections_without_parents.append(zotero_col)

    for zotero_col in collections_without_parents + collections_with_parents:
        try:
            col_key = zotero_col["key"]
            col_data = zotero_col["data"]
            col_name = col_data["name"]
            parent_collection_key = col_data.get("parentCollection")

            parent_collection = None
            if parent_collection_key and parent_collection_key in collection_map:
                parent_collection = collection_map[parent_collection_key]

            if library_type == "group":
                full_name = f"Zotero Group: {col_name}"
            else:
                full_name = f"Zotero: {col_name}"

            collection, created = Collection.objects.get_or_create(
                name=full_name,
                parent=parent_collection,
                defaults={},
            )

            CollectionPermission.objects.get_or_create(
                user=user,
                collection=collection,
                defaults={"permission": "MANAGE"},
            )

            collection_map[col_key] = collection

            if created:
                collections_created += 1
                logger.info(
                    "Created collection: %s (parent: %s)",
                    full_name,
                    parent_collection.name if parent_collection else "None",
                )
            else:
                collections_updated += 1
                logger.debug("Collection already exists: %s", full_name)

        except Exception as e:
            logger.error(
                "Error syncing collection %s: %s",
                zotero_col.get("key", "unknown"),
                str(e),
            )
            errors += 1

    return collections_created, collections_updated, errors


__all__ = ["sync_collections_with_hierarchy"]
