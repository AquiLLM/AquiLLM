"""
Celery tasks for Zotero synchronization.

Compat entrypoint: implementation lives in apps.integrations.zotero.tasks.
"""
from apps.integrations.zotero.tasks import sync_zotero_library

__all__ = ["sync_zotero_library"]
