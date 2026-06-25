"""WebSocket URL patterns for ingestion (primary runtime path)."""
from django.urls import re_path

from .consumers import IngestMonitorConsumer, IngestionDashboardConsumer

websocket_urlpatterns = [
    re_path(
        r"ingest/monitor/(?P<doc_id>[0-9a-f-]{36})/$",
        IngestMonitorConsumer.as_asgi(),
    ),
    re_path(r"ingest/dashboard/$", IngestionDashboardConsumer.as_asgi()),
]
