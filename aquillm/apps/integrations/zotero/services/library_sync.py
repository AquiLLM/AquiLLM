"""Orchestrate a full Zotero library sync (collections, downloads, PDF saves on main thread)."""
from __future__ import annotations

import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.utils import timezone

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import PDFDocument
from apps.integrations.zotero.models import ZoteroConnection

from aquillm.zotero_client import ZoteroAPIClient

logger = structlog.stdlib.get_logger(__name__)


def run_zotero_library_sync(user_id: int, library_config: Optional[dict] = None) -> dict:
    """Sync selected Zotero libraries for a user; returns stats dict."""
    if library_config is None:
        library_config = {"personal": ["ALL"], "groups": "all"}

    user = User.objects.get(id=user_id)
    connection = ZoteroConnection.objects.get(user=user)

    logger.info("Starting Zotero sync for user %s (ID: %s)", user.username, user_id)

    client = ZoteroAPIClient(
        api_key=connection.api_key,
        user_id=connection.zotero_user_id,
    )

    stats = {
        "collections_created": 0,
        "collections_updated": 0,
        "items_synced": 0,
        "pdfs_downloaded": 0,
        "errors": 0,
    }

    collection_map = {}

    logger.info("Processing library config: %s", library_config)

    for library_id, selected_collection_keys in library_config.items():
        if library_id == "personal":
            library_type = "user"
            group_id = None
            library_name = "Personal Library"
        else:
            library_type = "group"
            group_id = library_id
            all_groups = client.get_user_groups()
            group = next((g for g in all_groups if str(g["id"]) == library_id), None)
            library_name = group["data"]["name"] if group else f"Group {library_id}"

        logger.info("Syncing from %s...", library_name)

        all_collections = client.get_collections(group_id=group_id)

        if "ALL" in selected_collection_keys:
            selected_collections = all_collections
            logger.info("Syncing ALL collections (%s total)", len(all_collections))
        else:
            selected_collections = [
                col for col in all_collections if col["key"] in selected_collection_keys
            ]
            logger.info("Syncing %s selected collection(s)", len(selected_collections))

        collections_to_sync = set()
        for col in selected_collections:
            collections_to_sync.add(col["key"])
            parent_key = col["data"].get("parentCollection")
            while parent_key:
                collections_to_sync.add(parent_key)
                parent_col = next((c for c in all_collections if c["key"] == parent_key), None)
                if parent_col:
                    parent_key = parent_col["data"].get("parentCollection")
                else:
                    break

        filtered_collections = [col for col in all_collections if col["key"] in collections_to_sync]

        for col in filtered_collections:
            try:
                col_key = col["key"]
                col_data = col["data"]
                col_name = col_data["name"]
                parent_collection_key = col_data.get("parentCollection")

                parent_collection = None
                if parent_collection_key and parent_collection_key in collection_map:
                    parent_collection = collection_map[parent_collection_key]

                if library_type == "group":
                    full_name = f"Zotero ({library_name}): {col_name}"
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
                    stats["collections_created"] += 1
                    logger.info("Created collection: %s", full_name)
                else:
                    stats["collections_updated"] += 1

            except Exception as e:
                logger.error(
                    "Error syncing collection %s: %s",
                    col.get("key", "unknown"),
                    str(e),
                )
                stats["errors"] += 1

        items = client.get_top_level_items(group_id=group_id)

        items_to_sync = []
        for item in items:
            item_collections = item["data"].get("collections", [])
            if "ALL" in selected_collection_keys or any(
                col_key in collections_to_sync for col_key in item_collections
            ):
                items_to_sync.append(item)

        logger.info("Fetching children for %s items...", len(items_to_sync))
        item_children_map = {}
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_item = {
                executor.submit(client.get_item_children, item["key"], group_id=group_id): item
                for item in items_to_sync
            }
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    item_children_map[item["key"]] = future.result()
                except Exception as e:
                    logger.error("Error fetching children for %s: %s", item["key"], e)
                    item_children_map[item["key"]] = []

        pdf_attachments = []
        for item in items_to_sync:
            children = item_children_map.get(item["key"], [])
            for child in children:
                child_data = child["data"]
                if child_data.get("itemType") == "attachment" and child_data.get(
                    "contentType"
                ) == "application/pdf":
                    attachment_key = child["key"]
                    if not PDFDocument.objects.filter(zotero_item_key=attachment_key).exists():
                        pdf_attachments.append((item, child))

        logger.info("Downloading %s PDFs...", len(pdf_attachments))
        pdf_content_map = {}
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_attachment = {
                executor.submit(client.download_file, child["key"], group_id=group_id): (
                    item,
                    child,
                )
                for item, child in pdf_attachments
            }
            for future in as_completed(future_to_attachment):
                item, child = future_to_attachment[future]
                try:
                    content = future.result()
                    if content:
                        pdf_content_map[(item["key"], child["key"])] = content
                    else:
                        logger.warning("Could not download PDF for attachment %s", child["key"])
                except Exception as e:
                    logger.error("Error downloading %s: %s", child["key"], e)

        unfiled_collection = None
        if any(
            not (
                item["data"].get("collections", [])
                and item["data"].get("collections", [])[0] in collection_map
            )
            for item, child in pdf_attachments
            if pdf_content_map.get((item["key"], child["key"]))
        ):
            if library_type == "group":
                default_name = f"Zotero ({library_name}): Unfiled"
            else:
                default_name = "Zotero: Unfiled"
            unfiled_collection, _ = Collection.objects.get_or_create(
                name=default_name,
                parent=None,
                defaults={},
            )
            CollectionPermission.objects.get_or_create(
                user=user,
                collection=unfiled_collection,
                defaults={"permission": "MANAGE"},
            )

        def save_pdf_document(item, child):
            content = pdf_content_map.get((item["key"], child["key"]))
            if not content:
                return None

            item_data = item["data"]
            child_data = child["data"]
            item_collections = item_data.get("collections", [])
            attachment_key = child["key"]

            if item_collections and item_collections[0] in collection_map:
                target_collection = collection_map[item_collections[0]]
            else:
                target_collection = unfiled_collection

            title = item_data.get("title", "Untitled")
            filename = child_data.get("filename", f"{attachment_key}.pdf")

            pdf_doc = PDFDocument(
                title=title,
                collection=target_collection,
                ingested_by=user,
                full_text="",
                zotero_item_key=attachment_key,
            )
            pdf_doc.pdf_file.save(filename, ContentFile(content), save=False)
            pdf_doc.save()
            return title

        logger.info("Saving %s PDF documents (main thread; no threaded ORM)...", len(pdf_attachments))
        for item, child in pdf_attachments:
            try:
                title = save_pdf_document(item, child)
                if title:
                    stats["pdfs_downloaded"] += 1
                    logger.info("Created document: %s", title)
            except Exception as e:
                logger.error("Error saving document for %s: %s", child["key"], e)
                stats["errors"] += 1

        stats["items_synced"] += len(items_to_sync)

    connection.last_synced_at = timezone.now()
    connection.save()

    logger.info("Cleaning up empty collections...")
    empty_collections_deleted = 0
    for collection in collection_map.values():
        if not collection.documents:
            has_content = any(child.documents for child in collection.get_all_children())
            if not has_content:
                logger.info("Deleting empty collection: %s", collection.name)
                collection.delete()
                empty_collections_deleted += 1

    stats["empty_collections_deleted"] = empty_collections_deleted

    logger.info("Zotero sync completed for user %s. Stats: %s", user.username, stats)
    return stats


__all__ = ["run_zotero_library_sync"]
