import json
import structlog
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

logger = structlog.stdlib.get_logger(__name__)

class CrawlStatusConsumer(AsyncWebsocketConsumer):
    """
    Handles WebSocket connections for real-time crawl status updates.
    Connects users to a group based on their user ID.
    """
    async def connect(self):
        self.user = self.scope.get("user")

        if self.user is None or not self.user.is_authenticated:
            logger.warning("obs.crawl.ws_unauthenticated")
            await self.close()
            return

        self.group_name = f'crawl-status-{self.user.id}'
        logger.info("obs.crawl.ws_connecting", user_id=self.user.id, group_name=self.group_name)

        # Join user-specific group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()
        logger.info("obs.crawl.ws_connected", user_id=self.user.id, group_name=self.group_name)

    async def disconnect(self, close_code):
        if hasattr(self, 'group_name'):
            logger.info("obs.crawl.ws_disconnecting", user_id=self.user.id, group_name=self.group_name)
            # Leave user-specific group
            await self.channel_layer.group_discard(
                self.group_name,
                self.channel_name
            )
        else:
             logger.info("obs.crawl.ws_disconnect_unconnected")


    # Receive message from WebSocket (currently not expecting client messages)
    async def receive(self, text_data):
        # We don't expect messages from the client for this consumer,
        # but we include the method for completeness.
        logger.debug("obs.crawl.ws_unexpected_message", group_name=self.group_name, text_data=text_data)
        pass

    # --- Handlers for messages sent from the backend (Celery task) ---

    async def crawl_task_update(self, event):
        """
        Sends crawl status updates (start, progress, error, success) to the client.
        """
        message_data = event.get('data', {})
        message_type = event.get('type') # e.g., 'crawl.start', 'crawl.progress'

        logger.debug("obs.crawl.ws_send", message_type=message_type, group_name=self.group_name, message_data=message_data)

        # Send message to WebSocket client
        await self.send(text_data=json.dumps({
            'type': message_type,
            'payload': message_data,
        }))
        logger.debug("obs.crawl.ws_sent", message_type=message_type, channel_name=self.channel_name)