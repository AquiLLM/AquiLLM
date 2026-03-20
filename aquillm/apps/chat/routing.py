"""WebSocket routing for chat app."""
from django.urls import re_path

from .consumers import ChatConsumer

websocket_urlpatterns = [
    re_path(r"ws/convo/(?P<convo_id>[0-9]+)/$", ChatConsumer.as_asgi()),
]
