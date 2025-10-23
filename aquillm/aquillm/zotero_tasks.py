"""
Celery tasks for Zotero synchronization
"""
from celery import shared_task
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.utils import timezone
from django.db import transaction

from .models import ZoteroConnection, Collection, PDFDocument, CollectionPermission
from .zotero_client import ZoteroAPIClient

import logging
import io

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def sync_zotero_library(self, user_id: int):
    """
    Background task to sync a user's Zotero library.

    Process:
    1. Fetch collections from Zotero and create/update in AquiLLM
    2. Fetch items from Zotero
    3. For each item with PDF attachment:
       - Check if already synced (via zotero_item_key)
       - Download PDF
       - Create PDFDocument in appropriate collection
    4. Update last_sync_version

    Args:
        user_id: ID of the user to sync
    """
    try:
        # Get user and Zotero connection
        user = User.objects.get(id=user_id)
        connection = ZoteroConnection.objects.get(user=user)

        logger.info(f"Starting Zotero sync for user {user.username} (ID: {user_id})")

        # Initialize API client
        client = ZoteroAPIClient(
            api_key=connection.api_key,
            user_id=connection.zotero_user_id
        )

        # Track statistics
        stats = {
            'collections_created': 0,
            'collections_updated': 0,
            'items_synced': 0,
            'pdfs_downloaded': 0,
            'errors': 0
        }

        # Step 1: Sync collections
        logger.info("Syncing Zotero collections...")
        zotero_collections = client.get_collections(since_version=connection.last_sync_version)
        collection_map = {}  # Map Zotero key -> AquiLLM Collection

        for zotero_col in zotero_collections:
            try:
                col_key = zotero_col['key']
                col_name = zotero_col['data']['name']
                col_version = zotero_col['version']

                # Create or get collection in AquiLLM
                collection, created = Collection.objects.get_or_create(
                    name=f"Zotero: {col_name}",
                    defaults={'parent': None}
                )

                # Ensure user has MANAGE permission on this collection
                CollectionPermission.objects.get_or_create(
                    user=user,
                    collection=collection,
                    defaults={'permission': 'MANAGE'}
                )

                collection_map[col_key] = collection

                if created:
                    stats['collections_created'] += 1
                    logger.info(f"Created collection: {col_name}")
                else:
                    stats['collections_updated'] += 1
                    logger.info(f"Updated collection: {col_name}")

            except Exception as e:
                logger.error(f"Error syncing collection {zotero_col.get('key', 'unknown')}: {str(e)}")
                stats['errors'] += 1

        # Step 2: Sync items
        logger.info("Syncing Zotero items...")
        items, latest_version = client.get_top_level_items(since_version=connection.last_sync_version)

        for item in items:
            try:
                item_key = item['key']
                item_data = item['data']
                item_type = item_data.get('itemType', '')

                # Only process items that might have attachments
                if item_type in ['note', 'attachment']:
                    continue

                # Get item title
                title = item_data.get('title', 'Untitled')

                # Determine which collection to put this in
                # Items can belong to multiple collections, we'll use the first one
                collections_in_item = item_data.get('collections', [])
                target_collection = None

                if collections_in_item:
                    # Use first collection if it exists in our map
                    first_col_key = collections_in_item[0]
                    target_collection = collection_map.get(first_col_key)

                if not target_collection:
                    # Create a default "Zotero: Uncategorized" collection
                    target_collection, _ = Collection.objects.get_or_create(
                        name="Zotero: Uncategorized",
                        defaults={'parent': None}
                    )
                    CollectionPermission.objects.get_or_create(
                        user=user,
                        collection=target_collection,
                        defaults={'permission': 'MANAGE'}
                    )

                # Check if we've already synced this item
                existing_doc = PDFDocument.objects.filter(
                    zotero_item_key=item_key,
                    collection=target_collection
                ).first()

                if existing_doc:
                    logger.debug(f"Item {item_key} already synced, skipping")
                    continue

                # Step 3: Get children (attachments)
                children = client.get_item_children(item_key)

                # Look for PDF attachments
                for child in children:
                    child_data = child['data']
                    if child_data.get('itemType') == 'attachment':
                        content_type = child_data.get('contentType', '')
                        if 'pdf' in content_type.lower() or child_data.get('filename', '').endswith('.pdf'):
                            attachment_key = child['key']

                            # Check if this specific attachment was already synced
                            if PDFDocument.objects.filter(zotero_item_key=attachment_key).exists():
                                logger.debug(f"PDF attachment {attachment_key} already synced")
                                continue

                            # Download the PDF
                            logger.info(f"Downloading PDF: {title}")
                            pdf_content = client.download_file(attachment_key)

                            if pdf_content:
                                # Create PDFDocument
                                filename = child_data.get('filename', f"{title}.pdf")
                                pdf_file = ContentFile(pdf_content, name=filename)

                                with transaction.atomic():
                                    pdf_doc = PDFDocument(
                                        title=title,
                                        collection=target_collection,
                                        ingested_by=user,
                                        full_text='',  # Will be extracted on save
                                        zotero_item_key=attachment_key  # Store attachment key
                                    )
                                    pdf_doc.pdf_file.save(filename, pdf_file, save=False)
                                    pdf_doc.save()  # This will trigger text extraction

                                stats['pdfs_downloaded'] += 1
                                logger.info(f"Created PDFDocument for: {title}")

                stats['items_synced'] += 1

            except Exception as e:
                logger.error(f"Error syncing item {item.get('key', 'unknown')}: {str(e)}")
                stats['errors'] += 1

        # Update connection with latest sync info
        connection.last_sync_version = latest_version
        connection.last_synced_at = timezone.now()
        connection.save()

        logger.info(f"Zotero sync completed for user {user.username}. Stats: {stats}")
        return stats

    except User.DoesNotExist:
        logger.error(f"User {user_id} not found")
        raise
    except ZoteroConnection.DoesNotExist:
        logger.error(f"No Zotero connection for user {user_id}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during Zotero sync: {str(e)}")
        raise
