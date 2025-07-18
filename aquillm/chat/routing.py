from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/convo/(?P<convo_id>[0-9]+)/$", consumers.ChatConsumer.as_asgi()),
]