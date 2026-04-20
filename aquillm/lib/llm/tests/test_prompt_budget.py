"""Shared prompt budget policy."""
from __future__ import annotations

from unittest.mock import patch

from django.test import override_settings

from lib.llm.utils import prompt_budget as pb


@override_settings(TOKEN_EFFICIENCY_ENABLED=False, CONTEXT_PACKER_ENABLED=False)
def test_disabled_no_op():
    msgs = [{"role": "user", "content": "hello " * 200}]
    changed, new_max = pb.apply_preflight_trim_to_message_dicts("sys", msgs, 1024)
    assert changed is False
    assert new_max == 1024


@override_settings(
    TOKEN_EFFICIENCY_ENABLED=True,
    PROMPT_BUDGET_CONTEXT_LIMIT=800,
    PROMPT_BUDGET_SLACK_TOKENS=32,
    CONTEXT_PACKER_ENABLED=True,
)
@patch(
    "lib.llm.utils.context_packer.pack_messages_for_budget",
    wraps=__import__("lib.llm.utils.context_packer", fromlist=["pack_messages_for_budget"]).pack_messages_for_budget,
)
def test_preflight_invokes_context_packer_when_enabled(mock_pack):
    long = "word " * 1200
    msgs = [{"role": "user", "content": long}]
    pb.apply_preflight_trim_to_message_dicts("short", msgs, 256)
    assert mock_pack.called


@override_settings(
    TOKEN_EFFICIENCY_ENABLED=True,
    PROMPT_BUDGET_CONTEXT_LIMIT=800,
    PROMPT_BUDGET_SLACK_TOKENS=32,
    CONTEXT_PACKER_ENABLED=True,
)
@patch("lib.llm.utils.context_packer.pack_messages_for_budget", side_effect=RuntimeError("packer boom"))
def test_preflight_fails_open_when_packer_errors(_mock_pack):
    long = "word " * 1200
    msgs = [{"role": "user", "content": long}]
    changed, new_max = pb.apply_preflight_trim_to_message_dicts("short", msgs, 256)
    assert isinstance(new_max, int)
    assert changed is True
    assert len(msgs[0]["content"]) < len(long)


@override_settings(
    TOKEN_EFFICIENCY_ENABLED=False,
    PROMPT_BUDGET_CONTEXT_LIMIT=800,
    CONTEXT_PACKER_ENABLED=True,
    PROMPT_BUDGET_SLACK_TOKENS=32,
    CONTEXT_PIN_LAST_TURNS=1,
)
def test_packer_runs_without_token_efficiency_when_enabled():
    old = ("ZZOLDZZ " * 800).strip() + " "
    msgs = [
        {"role": "user", "content": old},
        {"role": "user", "content": "latest question about plants"},
    ]
    changed, new_max = pb.apply_preflight_trim_to_message_dicts("short", msgs, 256)
    assert changed is True
    assert "latest question" in str(msgs)
    assert "ZZOLDZZ" not in str(msgs)
    assert new_max == 256


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
