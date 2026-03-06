from typing import Awaitable
import uuid # Needed to convert the doc_id URL parameter string into a UUID for Document.get_by_id
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.auth import AuthMiddlewareStack
from channels.db import database_sync_to_async, aclose_old_connections

from django.contrib.auth.models import User
from django.apps import apps
from json import dumps
from aquillm.settings import DEBUG
from aquillm.models import DESCENDED_FROM_DOCUMENT, Document # Import Document so we can check ingestion_complete on connect
from functools import reduce
import logging
logger = logging.getLogger(__name__)

class IngestMonitorConsumer(AsyncWebsocketConsumer):
    # async def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     assert (self.channel_layer is not None and
    #         hasattr(self.channel_layer, 'group_send') and
    #         isinstance(self.channel_layer.group_send, Awaitable))# keeps type checker happy

    async def connect(self):
        self.user = self.scope.get('user', None)
        is_authenticated = bool(self.user and getattr(self.user, 'is_authenticated', False))
        if is_authenticated:
            await self.accept()
        else:
            await self.close()
        await self.channel_layer.group_add(f"document-ingest-{self.scope['url_route']['kwargs']['doc_id']}", self.channel_name) # type: ignore
        doc_id = self.scope['url_route']['kwargs']['doc_id'] # Get the document ID from the URL route
        doc = await database_sync_to_async(Document.get_by_id)(uuid.UUID(doc_id)) # Look up the document in the DB (must be async-safe)
        if doc and doc.ingestion_complete: # If ingestion already finished before the client connected, notify immediately
            await self.send(text_data=dumps({'type': 'document.ingest.complete', 'complete': True})) # Send complete so the frontend bar jumps to 100% instead of staying at 0%

    async def document_ingest_complete(self, event):
        await self.send(text_data=dumps(event))
        
    async def document_ingest_progress(self, event):
        await self.send(text_data=dumps(event))




class IngestionDashboardConsumer(AsyncWebsocketConsumer):
    @database_sync_to_async
    def __get_in_progress(self, user):
        querysets = [t.objects.filter(ingested_by=user, ingestion_complete=False).order_by('ingestion_date') for t in DESCENDED_FROM_DOCUMENT]    
        return reduce(lambda l,r : list(l) + list(r), querysets)
    
    async def connect(self):
        self.user = self.scope.get('user')
        is_authenticated = bool(self.user and getattr(self.user, 'is_authenticated', False))
        if is_authenticated:
            await self.accept()
        else:
            await self.close()
        await self.channel_layer.group_add(f"ingestion-dashboard-{self.user.id}", self.channel_name) # type: ignore
        in_progress = await self.__get_in_progress(self.user)
        for doc in in_progress:
            await self.send(dumps({'type': 'document.ingestion.start', 'documentName': doc.title, 'documentId': str(doc.id)}))

        
    async def document_ingestion_start(self, event):
        await self.send(text_data=dumps(event))
