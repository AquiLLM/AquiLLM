import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

logger = logging.getLogger(__name__)

class CrawlStatusConsumer(AsyncWebsocketConsumer):
    """
    Handles WebSocket connections for real-time crawl status updates.
    Connects users to a group based on their user ID.
    """
    async def connect(self):
        self.user = self.scope.get("user")

        if self.user is None or not self.user.is_authenticated:
            logger.warning("WebSocket connection attempt by unauthenticated user.")
            await self.close()
            return

        self.group_name = f'crawl-status-{self.user.id}'
        logger.info(f"User {self.user.id} connecting to WebSocket group {self.group_name}")

        # Join user-specific group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()
        logger.info(f"User {self.user.id} successfully connected to {self.group_name}")

    async def disconnect(self, close_code):
        if hasattr(self, 'group_name'):
            logger.info(f"User {self.user.id} disconnecting from WebSocket group {self.group_name}")
            # Leave user-specific group
            await self.channel_layer.group_discard(
                self.group_name,
                self.channel_name
            )
        else:
             logger.info("Unauthenticated or unconnected user disconnecting.")


    # Receive message from WebSocket (currently not expecting client messages)
    async def receive(self, text_data):
        # We don't expect messages from the client for this consumer,
        # but we include the method for completeness.
        logger.debug(f"Received unexpected message from client in {self.group_name}: {text_data}")
        pass

    # --- Handlers for messages sent from the backend (Celery task) ---

    async def crawl_task_update(self, event):
        """
        Sends crawl status updates (start, progress, error, success) to the client.
        """
        message_data = event.get('data', {})
        message_type = event.get('type') # e.g., 'crawl.start', 'crawl.progress'

        logger.debug(f"Sending '{message_type}' to group {self.group_name}: {message_data}")

        # Send message to WebSocket client
        await self.send(text_data=json.dumps({
            'type': message_type,
            'payload': message_data,
        }))
        logger.debug(f"Sent '{message_type}' to client {self.channel_name}")