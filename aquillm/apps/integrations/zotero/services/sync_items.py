"""Sync Zotero library items and PDF attachments into AquiLLM."""
from __future__ import annotations

import structlog

from django.contrib.auth.models import User
from django.core.files.base import ContentFile

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import PDFDocument

from aquillm.zotero_client import ZoteroAPIClient

logger = structlog.stdlib.get_logger(__name__)


def sync_items_from_library(
    client: ZoteroAPIClient,
    user: User,
    collection_map: dict,
    library_id=None,
    library_type: str = "user",
):
    """
    Sync items from a library (personal or group).

    Returns:
        Tuple of (items_synced, pdfs_downloaded, errors)
    """
    items_synced = 0
    pdfs_downloaded = 0
    errors = 0

    logger.info("obs.zotero.sync_items_start", library_type=library_type)
    items = client.get_top_level_items(group_id=library_id)

    for item in items:
        try:
            item_key = item["key"]
            item_data = item["data"]
            item_type = item_data.get("itemType", "")

            if item_type in ["note", "attachment"]:
                continue

            title = item_data.get("title", "Untitled")

            collections_in_item = item_data.get("collections", [])
            target_collection = None

            if collections_in_item:
                first_col_key = collections_in_item[0]
                target_collection = collection_map.get(first_col_key)

            if not target_collection:
                uncategorized_name = f"Zotero {library_type.capitalize()}: Uncategorized"
                target_collection, _ = Collection.objects.get_or_create(
                    name=uncategorized_name,
                    defaults={"parent": None},
                )
                CollectionPermission.objects.get_or_create(
                    user=user,
                    collection=target_collection,
                    defaults={"permission": "MANAGE"},
                )

            existing_doc = PDFDocument.objects.filter(
                zotero_item_key=item_key,
                collection=target_collection,
            ).first()

            if existing_doc:
                logger.debug("obs.zotero.sync_item_skip", item_key=item_key)
                continue

            children = client.get_item_children(item_key, group_id=library_id)

            for child in children:
                child_data = child["data"]
                if child_data.get("itemType") == "attachment":
                    content_type = child_data.get("contentType", "")
                    if "pdf" in content_type.lower() or child_data.get("filename", "").endswith(".pdf"):
                        attachment_key = child["key"]

                        if PDFDocument.objects.filter(zotero_item_key=attachment_key).exists():
                            logger.debug("obs.zotero.sync_item_skip", item_key=attachment_key)
                            continue

                        logger.info("obs.zotero.download_file", title=title)
                        pdf_content = client.download_file(attachment_key, group_id=library_id)

                        if pdf_content:
                            filename = child_data.get("filename", f"{title}.pdf")
                            pdf_file = ContentFile(pdf_content, name=filename)

                            pdf_doc = PDFDocument(
                                title=title,
                                collection=target_collection,
                                ingested_by=user,
                                full_text="",
                                zotero_item_key=attachment_key,
                            )
                            pdf_doc.pdf_file.save(filename, pdf_file, save=False)
                            pdf_doc.save()

                            pdfs_downloaded += 1
                            logger.info("obs.zotero.sync_document_created", title=title)

            items_synced += 1

        except Exception as e:
            logger.error(
                "obs.zotero.sync_item_error",
                item_key=item.get("key", "unknown"),
                error=str(e),
            )
            errors += 1

    return items_synced, pdfs_downloaded, errors


__all__ = ["sync_items_from_library"]
