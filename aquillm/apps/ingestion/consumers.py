"""WebSocket consumers for ingestion monitoring (primary runtime path)."""
from __future__ import annotations

import structlog
import uuid
from functools import reduce
from json import dumps

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from apps.documents.models import DESCENDED_FROM_DOCUMENT, Document
from apps.documents.services.document_meta import (
    document_has_raw_media,
    document_modality,
    document_provider_model,
    document_provider_name,
)

logger = structlog.stdlib.get_logger(__name__)


class IngestMonitorConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope.get("user", None)
        is_authenticated = bool(self.user and getattr(self.user, "is_authenticated", False))
        if not is_authenticated:
            await self.close()
            return
        await self.accept()
        await self.channel_layer.group_add(
            f"document-ingest-{self.scope['url_route']['kwargs']['doc_id']}", self.channel_name
        )  # type: ignore
        doc_id = self.scope["url_route"]["kwargs"]["doc_id"]
        doc = await database_sync_to_async(Document.get_by_id)(uuid.UUID(doc_id))
        if doc and doc.ingestion_complete:
            await self.send(
                text_data=dumps({"type": "document.ingest.complete", "complete": True})
            )

    async def document_ingest_complete(self, event):
        await self.send(text_data=dumps(event))

    async def document_ingest_progress(self, event):
        await self.send(text_data=dumps(event))


class IngestionDashboardConsumer(AsyncWebsocketConsumer):
    @database_sync_to_async
    def __get_in_progress(self, user):
        querysets = [
            t.objects.filter(ingested_by=user, ingestion_complete=False).order_by("ingestion_date")
            for t in DESCENDED_FROM_DOCUMENT
        ]
        return reduce(lambda l, r: list(l) + list(r), querysets)

    async def connect(self):
        self.user = self.scope.get("user")
        is_authenticated = bool(self.user and getattr(self.user, "is_authenticated", False))
        if not is_authenticated:
            await self.close()
            return
        await self.accept()
        await self.channel_layer.group_add(
            f"ingestion-dashboard-{self.user.id}", self.channel_name
        )  # type: ignore
        in_progress = await self.__get_in_progress(self.user)
        for doc in in_progress:
            await self.send(
                dumps(
                    {
                        "type": "document.ingestion.start",
                        "documentName": doc.title,
                        "documentId": str(doc.id),
                        "modality": document_modality(doc),
                        "rawMediaSaved": document_has_raw_media(doc),
                        "textExtracted": bool((doc.full_text or "").strip()),
                        "provider": document_provider_name(doc),
                        "providerModel": document_provider_model(doc),
                    }
                )
            )

    async def document_ingestion_start(self, event):
        await self.send(text_data=dumps(event))


__all__ = ["IngestMonitorConsumer", "IngestionDashboardConsumer"]
