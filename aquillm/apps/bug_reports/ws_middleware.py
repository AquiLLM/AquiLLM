"""WebSocket ASGI middleware for activity logging.

Wraps the Channels router to log connect/receive events to Redis
without modifying any individual consumer.
"""
from .activity import async_log_activity, now_iso


class BugReportWSMiddleware:
    """ASGI middleware that logs WebSocket events to the activity tracker."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "websocket":
            return await self.app(scope, receive, send)

        user = scope.get("user")
        if not user or not getattr(user, "is_authenticated", False):
            return await self.app(scope, receive, send)

        user_id = user.id
        path = scope.get("path", "")

        # Log the connection
        await async_log_activity(user_id, {
            "type": "ws_connect",
            "path": path,
            "timestamp": now_iso(),
        })

        # Wrap receive to log incoming messages
        original_receive = receive

        async def logging_receive():
            message = await original_receive()
            if message.get("type") == "websocket.receive":
                await async_log_activity(user_id, {
                    "type": "ws_message",
                    "path": path,
                    "timestamp": now_iso(),
                })
            elif message.get("type") == "websocket.disconnect":
                await async_log_activity(user_id, {
                    "type": "ws_disconnect",
                    "path": path,
                    "timestamp": now_iso(),
                })
            return message

        return await self.app(scope, logging_receive, send)
