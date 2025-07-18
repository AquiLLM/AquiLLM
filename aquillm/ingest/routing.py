# type: ignore
from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r"ingest/monitor/(?P<doc_id>[0-9a-f-]{36})/$", consumers.IngestMonitorConsumer.as_asgi()),
    re_path(r"ingest/dashboard/$", consumers.IngestionDashboardConsumer.as_asgi()),
]