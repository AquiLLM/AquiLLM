"""Websocket auth: unauthenticated clients must not join channel groups."""
import uuid
from unittest.mock import AsyncMock

import pytest
from django.contrib.auth.models import AnonymousUser
from django.contrib.auth import get_user_model

from ingest.consumers import IngestMonitorConsumer, IngestionDashboardConsumer

User = get_user_model()


@pytest.mark.asyncio
async def test_monitor_consumer_unauthenticated_skips_group_add():
    consumer = IngestMonitorConsumer()
    consumer.base_send = AsyncMock()
    doc_id = str(uuid.uuid4())
    consumer.scope = {
        "user": AnonymousUser(),
        "url_route": {"kwargs": {"doc_id": doc_id}},
    }
    consumer.channel_name = "test.channel"
    layer = AsyncMock()
    consumer.channel_layer = layer

    await consumer.connect()

    layer.group_add.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.django_db
async def test_monitor_consumer_authenticated_joins_group():
    user = User.objects.create_user(username="ingest_mon", password="pw")
    consumer = IngestMonitorConsumer()
    consumer.base_send = AsyncMock()
    doc_id = str(uuid.uuid4())
    consumer.scope = {
        "user": user,
        "url_route": {"kwargs": {"doc_id": doc_id}},
    }
    consumer.channel_name = "test.channel"
    layer = AsyncMock()
    consumer.channel_layer = layer

    await consumer.connect()

    layer.group_add.assert_awaited_once_with(
        f"document-ingest-{doc_id}",
        "test.channel",
    )


@pytest.mark.asyncio
async def test_dashboard_consumer_unauthenticated_skips_group_add():
    consumer = IngestionDashboardConsumer()
    consumer.base_send = AsyncMock()
    consumer.scope = {"user": AnonymousUser()}
    consumer.channel_name = "test.channel"
    layer = AsyncMock()
    consumer.channel_layer = layer

    await consumer.connect()

    layer.group_add.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.django_db
async def test_dashboard_consumer_authenticated_joins_group():
    user = User.objects.create_user(username="dashauth", password="pw")
    consumer = IngestionDashboardConsumer()
    consumer.base_send = AsyncMock()
    consumer.scope = {"user": user}
    consumer.channel_name = "test.channel"
    layer = AsyncMock()
    consumer.channel_layer = layer

    await consumer.connect()

    layer.group_add.assert_awaited_once_with(
        f"ingestion-dashboard-{user.id}",
        "test.channel",
    )
