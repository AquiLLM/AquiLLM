"""Celery tasks for Zotero synchronization."""
from __future__ import annotations

import logging
from typing import Optional

from celery import shared_task
from django.contrib.auth.models import User

from apps.integrations.zotero.models import ZoteroConnection
from apps.integrations.zotero.services.library_sync import run_zotero_library_sync

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="aquillm.zotero_tasks.sync_zotero_library")
def sync_zotero_library(self, user_id: int, library_config: Optional[dict] = None):
    """
    Background task to sync a user's Zotero library including personal and group libraries.

    See ``run_zotero_library_sync`` for process details.
    """
    try:
        return run_zotero_library_sync(user_id, library_config)
    except User.DoesNotExist:
        logger.error("User %s not found", user_id)
        raise
    except ZoteroConnection.DoesNotExist:
        logger.error("No Zotero connection for user %s", user_id)
        raise
    except Exception as e:
        logger.error("Unexpected error during Zotero sync: %s", str(e))
        raise


__all__ = ["sync_zotero_library"]
