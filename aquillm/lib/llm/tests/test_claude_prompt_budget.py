"""Claude path applies shared prompt budget when enabled."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from django.test import override_settings

from lib.llm.providers.claude import ClaudeInterface


@pytest.mark.asyncio
@override_settings(
    TOKEN_EFFICIENCY_ENABLED=True,
    PROMPT_BUDGET_CONTEXT_LIMIT=800,
    PROMPT_BUDGET_SLACK_TOKENS=32,
)
async def test_get_message_preflight_trims_long_user_content():
    resp = MagicMock()
    resp.content = [SimpleNamespace(text="done")]
    resp.stop_reason = "end_turn"
    resp.usage = MagicMock(input_tokens=1, output_tokens=1)

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=resp)

    iface = ClaudeInterface(client)
    long = "word " * 1200
    msgs = [{"role": "user", "content": long}]
    await iface.get_message(
        **(iface.base_args | {"system": "sys", "messages": msgs, "max_tokens": 128}),
    )
    sent = client.messages.create.await_args.kwargs["messages"][0]["content"]
    assert len(sent) < len(long)
