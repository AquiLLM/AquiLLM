"""Shared prompt budget policy."""
from __future__ import annotations

from django.test import override_settings

from lib.llm.utils import prompt_budget as pb


@override_settings(TOKEN_EFFICIENCY_ENABLED=False)
def test_disabled_no_op():
    msgs = [{"role": "user", "content": "hello " * 200}]
    changed, new_max = pb.apply_preflight_trim_to_message_dicts("sys", msgs, 1024)
    assert changed is False
    assert new_max == 1024


def test_sync_trimmed_dicts_into_pydantic():
    from lib.llm.types.messages import UserMessage

    p = UserMessage(content="hello " * 50)
    trimmed = [{"role": "user", "content": "short"}]
    pb.sync_trimmed_dicts_into_pydantic_messages([p], trimmed)
    assert p.content == "short"


@override_settings(
    TOKEN_EFFICIENCY_ENABLED=True,
    PROMPT_BUDGET_CONTEXT_LIMIT=512,
    PROMPT_BUDGET_SLACK_TOKENS=32,
    PROMPT_BUDGET_MAX_TOKENS_CAP=8192,
)
def test_trims_when_over_budget():
    long = "word " * 2000
    msgs = [{"role": "user", "content": long}]
    changed, _ = pb.apply_preflight_trim_to_message_dicts("short", msgs, 256)
    assert changed is True
    assert len(msgs[0]["content"]) < len(long)
