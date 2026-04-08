"""Celery tasks for Zotero synchronization."""
from __future__ import annotations

import structlog
from typing import Optional

from celery import shared_task
from django.contrib.auth.models import User

from apps.integrations.zotero.models import ZoteroConnection
from apps.integrations.zotero.services.library_sync import run_zotero_library_sync

logger = structlog.stdlib.get_logger(__name__)


@shared_task(bind=True, name="aquillm.zotero_tasks.sync_zotero_library")
def sync_zotero_library(self, user_id: int, library_config: Optional[dict] = None):
    """
    Background task to sync a user's Zotero library including personal and group libraries.

    See ``run_zotero_library_sync`` for process details.
    """
    try:
        return run_zotero_library_sync(user_id, library_config)
    except User.DoesNotExist:
        logger.error("obs.zotero.task_user_not_found", user_id=user_id)
        raise
    except ZoteroConnection.DoesNotExist:
        logger.error("obs.zotero.task_no_connection", user_id=user_id)
        raise
    except Exception as e:
        logger.error("obs.zotero.task_error", error_type=type(e).__name__, error=str(e))
        raise


__all__ = ["sync_zotero_library"]
