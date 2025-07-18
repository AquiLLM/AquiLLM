from typing import Awaitable
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.auth import AuthMiddlewareStack
from channels.db import database_sync_to_async, aclose_old_connections

from django.contrib.auth.models import User
from django.apps import apps
from json import dumps
from aquillm.settings import DEBUG
from aquillm.models import DESCENDED_FROM_DOCUMENT
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
