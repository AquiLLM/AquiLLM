"""
Celery tasks for Zotero synchronization
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from celery import shared_task
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.utils import timezone
from typing import Optional, Literal

from pyzotero import zotero

from .models import ZoteroConnection, Collection, PDFDocument, CollectionPermission

import logging

logger = logging.getLogger(__name__)


# @shared_task(bind=True)
# def sync_collection(self, user: int | User, type: Literal["user", "group"], zotero_id: str, sync_items: bool = True):
#     if isinstance(user, int):
#         user = User.objects.get(id=user)
#     user_zotero_connection = ZoteroConnection.objects.get(user=user)
#     zotero_user_id = user_zotero_connection.zotero_user_id
#     api_key = user_zotero_connection.api_key
#     user_zot = zotero.Zotero(zotero_user_id, 'user', api_key)
#     stats = {
#         'collections_created': 0,
#         'collections_updated': 0,
#         'items_synced': 0,
#         'pdfs_downloaded': 0,
#         'errors': 0
#     }

@shared_task(bind=True)
def sync_zotero_library(self, user_id: int, library_config: Optional[dict] = None):
    """
    Background task to sync a user's Zotero library including personal and group libraries.

    Process:
    1. Fetch collections from selected libraries
    2. Filter collections based on user selection
    3. Fetch items from selected collections
    4. For each item with PDF attachment:
       - Check if already synced (via zotero_item_key)
       - Download PDF
       - Create PDFDocument in appropriate collection

    Args:
        user_id: ID of the user to sync
        library_config: Optional dict mapping library IDs to collection keys
                       Format: {'personal': ['ALL'] or ['key1', 'key2'], 'group_id': ['ALL'] or ['key1']}
                       If None, syncs all libraries (default behavior for backward compatibility)
    """
    # Default to syncing everything if no config provided
    if library_config is None:
        library_config = {'personal': ['ALL'], 'groups': 'all'}
    try:
        # Get user and Zotero connection
        user = User.objects.get(id=user_id)
        connection = ZoteroConnection.objects.get(user=user)

        logger.info(f"Starting Zotero sync for user {user.username} (ID: {user_id})")

        # Store API credentials for creating pyzotero clients
        api_key = connection.api_key
        user_id_zotero = connection.zotero_user_id

        # Create pyzotero client for user library (used for fetching groups)
        user_zot = zotero.Zotero(user_id_zotero, 'user', api_key)

        # Track statistics
        stats = {
            'collections_created': 0,
            'collections_updated': 0,
            'items_synced': 0,
            'pdfs_downloaded': 0,
            'errors': 0
        }

        # Map Zotero collection keys to AquiLLM Collections
        collection_map = {}

        # Step 1: Parse library configuration
        logger.info(f"Processing library config: {library_config}")

        # Helper function to filter collections based on selection
        def should_sync_collection(col_key, selected_keys):
            """Check if collection should be synced based on user selection"""
            return 'ALL' in selected_keys or col_key in selected_keys

        # Step 2: Process each library in the config
        for library_id, selected_collection_keys in library_config.items():
            if library_id == 'personal':
                library_type = 'user'
                library_name = "Personal Library"
                # Use user library client
                zot = zotero.Zotero(user_id_zotero, 'user', api_key)
            else:
                library_type = 'group'
                # Fetch group name
                all_groups = user_zot.everything(user_zot.groups())
                group = next((g for g in all_groups if str(g['id']) == library_id), None)
                library_name = group['data']['name'] if group else f"Group {library_id}"
                # Create group library client
                zot = zotero.Zotero(library_id, 'group', api_key)

            logger.info(f"Syncing from {library_name}...")

            # Fetch all collections from this library
            all_collections = zot.everything(zot.collections())

            # Filter collections based on user selection
            if 'ALL' in selected_collection_keys:
                selected_collections = all_collections
                logger.info(f"Syncing ALL collections ({len(all_collections)} total)")
            else:
                selected_collections = [col for col in all_collections if col['key'] in selected_collection_keys]
                logger.info(f"Syncing {len(selected_collections)} selected collection(s)")

            # Sync selected collections with hierarchy
            # We need to include parent collections even if not explicitly selected
            # to maintain hierarchy
            collections_to_sync = set()
            for col in selected_collections:
                collections_to_sync.add(col['key'])
                # Add all parent collections
                parent_key = col['data'].get('parentCollection')
                while parent_key:
                    collections_to_sync.add(parent_key)
                    parent_col = next((c for c in all_collections if c['key'] == parent_key), None)
                    if parent_col:
                        parent_key = parent_col['data'].get('parentCollection')
                    else:
                        break

            # Filter to collections we need to sync
            filtered_collections = [col for col in all_collections if col['key'] in collections_to_sync]

            # Sync collections
            for col in filtered_collections:
                try:
                    col_key = col['key']
                    col_data = col['data']
                    col_name = col_data['name']
                    parent_collection_key = col_data.get('parentCollection')

                    # Determine parent collection in AquiLLM
                    parent_collection = None
                    if parent_collection_key and parent_collection_key in collection_map:
                        parent_collection = collection_map[parent_collection_key]

                    # Create collection name with library prefix
                    if library_type == 'group':
                        full_name = f"Zotero ({library_name}): {col_name}"
                    else:
                        full_name = f"Zotero: {col_name}"

                    # Create or get collection in AquiLLM
                    collection, created = Collection.objects.get_or_create(
                        name=full_name,
                        parent=parent_collection,
                        defaults={}
                    )

                    # Ensure user has MANAGE permission
                    CollectionPermission.objects.get_or_create(
                        user=user,
                        collection=collection,
                        defaults={'permission': 'MANAGE'}
                    )

                    # Store in map
                    collection_map[col_key] = collection

                    if created:
                        stats['collections_created'] += 1
                        logger.info(f"Created collection: {full_name}")
                    else:
                        stats['collections_updated'] += 1

                except Exception as e:
                    logger.error(f"Error syncing collection {col.get('key', 'unknown')}: {str(e)}")
                    stats['errors'] += 1

            # Sync items from selected collections
            items = zot.everything(zot.top())

            # Filter items to only those in selected collections
            items_to_sync = []
            for item in items:
                item_collections = item['data'].get('collections', [])
                if 'ALL' in selected_collection_keys or any(col_key in collections_to_sync for col_key in item_collections):
                    items_to_sync.append(item)

            # Fetch children for all items in parallel
            logger.info(f"Fetching children for {len(items_to_sync)} items...")
            item_children_map = {}
            with ThreadPoolExecutor(max_workers=16) as executor:
                future_to_item = {
                    executor.submit(zot.children, item['key']): item
                    for item in items_to_sync
                }
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        item_children_map[item['key']] = future.result()
                    except Exception as e:
                        logger.error(f"Error fetching children for {item['key']}: {e}")
                        item_children_map[item['key']] = []

            # Identify all PDF attachments to download
            pdf_attachments = []  # List of (item, child) tuples
            for item in items_to_sync:
                children = item_children_map.get(item['key'], [])
                for child in children:
                    child_data = child['data']
                    if child_data.get('itemType') == 'attachment' and child_data.get('contentType') == 'application/pdf':
                        attachment_key = child['key']
                        if not PDFDocument.objects.filter(zotero_item_key=attachment_key).exists():
                            pdf_attachments.append((item, child))

            # Download all PDFs in parallel
            logger.info(f"Downloading {len(pdf_attachments)} PDFs...")
            pdf_content_map = {}
            with ThreadPoolExecutor(max_workers=16) as executor:
                future_to_attachment = {
                    executor.submit(zot.file, child['key']): (item, child)
                    for item, child in pdf_attachments
                }
                for future in as_completed(future_to_attachment):
                    item, child = future_to_attachment[future]
                    try:
                        content = future.result()
                        if content:
                            pdf_content_map[(item['key'], child['key'])] = content
                        else:
                            logger.warning(f"Could not download PDF for attachment {child['key']}")
                    except Exception as e:
                        logger.error(f"Error downloading {child['key']}: {e}")

            # Prepare default collection if needed (do this before parallel saves)
            unfiled_collection = None
            if any(
                not (item['data'].get('collections', []) and item['data'].get('collections', [])[0] in collection_map)
                for item, child in pdf_attachments
                if pdf_content_map.get((item['key'], child['key']))
            ):
                if library_type == 'group':
                    default_name = f"Zotero ({library_name}): Unfiled"
                else:
                    default_name = "Zotero: Unfiled"
                unfiled_collection, _ = Collection.objects.get_or_create(
                    name=default_name,
                    parent=None,
                    defaults={}
                )
                CollectionPermission.objects.get_or_create(
                    user=user,
                    collection=unfiled_collection,
                    defaults={'permission': 'MANAGE'}
                )

            def save_pdf_document(item, child):
                """Save a single PDF document (runs in thread pool)."""
                content = pdf_content_map.get((item['key'], child['key']))
                if not content:
                    return None

                item_data = item['data']
                child_data = child['data']
                item_collections = item_data.get('collections', [])
                attachment_key = child['key']

                # Determine target collection
                if item_collections and item_collections[0] in collection_map:
                    target_collection = collection_map[item_collections[0]]
                else:
                    target_collection = unfiled_collection

                title = item_data.get('title', 'Untitled')
                filename = child_data.get('filename', f'{attachment_key}.pdf')

                pdf_doc = PDFDocument(
                    title=title,
                    collection=target_collection,
                    ingested_by=user,
                    full_text='',
                    zotero_item_key=attachment_key
                )
                pdf_doc.pdf_file.save(filename, ContentFile(content), save=False)
                pdf_doc.save()
                return title

            # Create PDFDocuments in parallel
            logger.info(f"Saving {len(pdf_attachments)} PDF documents...")
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_child = {
                    executor.submit(save_pdf_document, item, child): child
                    for item, child in pdf_attachments
                }
                for future in as_completed(future_to_child):
                    child = future_to_child[future]
                    try:
                        title = future.result()
                        if title:
                            stats['pdfs_downloaded'] += 1
                            logger.info(f"Created document: {title}")
                    except Exception as e:
                        logger.error(f"Error saving document for {child['key']}: {e}")
                        stats['errors'] += 1

            stats['items_synced'] += len(items_to_sync)

        # Update connection with latest sync info
        # Note: We're using the personal library version as the baseline
        # Group libraries have their own versions but we track them together
        connection.last_synced_at = timezone.now()
        connection.save()

        # Cleanup: Remove empty collections that were created during sync
        logger.info("Cleaning up empty collections...")
        empty_collections_deleted = 0
        # for collection in collection_map.values():
        #     # Check if collection has any documents
        #     if not collection.documents:
        #         # Also check if it has any children with documents
        #         has_content = any(child.documents for child in collection.get_all_children())
        #         if not has_content:
        #             logger.info(f"Deleting empty collection: {collection.name}")
        #             collection.delete()
        #             empty_collections_deleted += 1

        stats['empty_collections_deleted'] = empty_collections_deleted

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
