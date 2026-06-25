"""Gemini path applies shared prompt budget when enabled."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from django.test import override_settings

from lib.llm.providers.gemini import GeminiInterface
from lib.llm.types.messages import UserMessage


@pytest.mark.asyncio
@override_settings(
    TOKEN_EFFICIENCY_ENABLED=True,
    PROMPT_BUDGET_CONTEXT_LIMIT=800,
    PROMPT_BUDGET_SLACK_TOKENS=32,
)
async def test_get_message_preflight_trims_parallel_pydantic():
    resp = MagicMock()
    resp.function_calls = None
    resp.usage_metadata = MagicMock(prompt_token_count=1, candidates_token_count=1)
    resp.text = "ok"

    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=resp)

    iface = GeminiInterface(client, model="gemini-2.5-flash")
    long = "word " * 1200
    msgs = [{"role": "user", "content": long}]
    pyd = [UserMessage(content=long)]
    await iface.get_message(
        system="sys",
        messages=msgs,
        messages_pydantic=pyd,
        max_tokens=128,
        tools=None,
        tool_choice=None,
        thinking_budget=None,
    )
    assert len(pyd[0].content) < len(long)
