"""Tests for best-effort WebSocket delivery after client disconnects."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apps.chat.consumers.chat_transport import best_effort_send


async def test_best_effort_send_swallows_recognized_disconnect_only():
    async def send(*, text_data):
        raise RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'"
        )

    consumer = SimpleNamespace(send=send, transport_connected=True)

    sent = await best_effort_send(consumer, text_data="{}")

    assert sent is False
    assert consumer.transport_connected is False


async def test_best_effort_send_skips_transport_already_marked_disconnected():
    async def send(*, text_data):
        pytest.fail("send must not run after transport disconnect")

    consumer = SimpleNamespace(send=send, transport_connected=False)

    assert await best_effort_send(consumer, text_data="{}") is False


async def test_best_effort_send_reraises_application_failure():
    async def send(*, text_data):
        raise ValueError("serialization bug")

    consumer = SimpleNamespace(send=send, transport_connected=True)

    with pytest.raises(ValueError, match="serialization bug"):
        await best_effort_send(consumer, text_data="{}")

    assert consumer.transport_connected is True
