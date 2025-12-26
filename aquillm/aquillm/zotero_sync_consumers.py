import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)


class ZoteroSyncConsumer(AsyncWebsocketConsumer):
    """
    Handles WebSocket connections for real-time Zotero sync status updates.
    Connects users to a group based on their user ID.
    """

    async def connect(self):
        self.user = self.scope.get("user")

        if self.user is None or not self.user.is_authenticated:
            logger.warning("Zotero sync WebSocket connection attempt by unauthenticated user.")
            await self.close()
            return

        self.group_name = f'zotero-sync-{self.user.id}'
        logger.info(f"User {self.user.id} connecting to Zotero sync WebSocket group {self.group_name}")

        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()
        logger.info(f"User {self.user.id} successfully connected to {self.group_name}")

    async def disconnect(self, close_code):
        if hasattr(self, 'group_name'):
            logger.info(f"User {self.user.id} disconnecting from Zotero sync WebSocket group {self.group_name}")
            await self.channel_layer.group_discard(
                self.group_name,
                self.channel_name
            )

    async def receive(self, text_data):
        # We don't expect messages from the client
        logger.debug(f"Received unexpected message from client in {self.group_name}: {text_data}")

    async def zotero_sync_update(self, event):
        """
        Sends Zotero sync status updates to the client.
        """
        message_data = event.get('data', {})
        message_type = event.get('message_type', 'info')

        logger.debug(f"Sending Zotero sync update to group {self.group_name}: {message_data}")

        await self.send(text_data=json.dumps({
            'type': message_type,
            'message': message_data.get('message', ''),
            'timestamp': message_data.get('timestamp'),
        }))
