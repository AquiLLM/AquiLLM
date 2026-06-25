"""Channel-layer notifications for ingest-monitor WebSocket clients."""
from __future__ import annotations

from typing import Any

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer


def notify_ingest_monitor_progress(doc_id: Any, progress: int) -> None:
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"document-ingest-{doc_id}",
        {"type": "document.ingest.progress", "progress": progress},
    )


def notify_ingest_monitor_complete(doc_id: Any, *, complete: bool = True) -> None:
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"document-ingest-{doc_id}",
        {"type": "document.ingest.complete", "complete": complete},
    )


__all__ = ["notify_ingest_monitor_progress", "notify_ingest_monitor_complete"]
