"""Preflight and live protected text requests use one provider payload shape."""

from types import SimpleNamespace

import pytest

from lib.llm.evidence_guard import EvidenceProtection, protect_evidence
from lib.llm.providers import openai_request as request


def provider():
    return SimpleNamespace(client=SimpleNamespace(base_url="http://vllm:8000/v1/"),
        base_args={"model": "local-research-model"})


def shape(**changes):
    values = dict(system_text="Grounding rules.",
        message_list=[{"role": "user", "content": "Selected evidence."}],
        max_tokens=123, thinking_budget=0)
    values.update(changes)
    function = getattr(request, "protected_text_request_arguments", None)
    return function(provider(), **values) if function else None


def test_protected_shape_keeps_exact_provider_configuration(monkeypatch):
    monkeypatch.setenv("OPENAI_SYSTEM_ROLE", "system")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.2")
    monkeypatch.setenv("OPENAI_TOP_P", "0.95")
    assert shape(response_format={"type": "json_object"}) == {
        "model": "local-research-model",
        "messages": [{"role": "system", "content": "Grounding rules."},
                     {"role": "user", "content": "Selected evidence."}],
        "max_tokens": 123,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "temperature": 0.2, "top_p": 0.95,
        "response_format": {"type": "json_object"},
    }


@pytest.mark.parametrize("change", [
    {"message_list": [{"role": "user", "content": [{"type": "image_url", "image_url": "private"}]}]},
    {"message_list": [{"role": "user", "content": "text", "name": "unbound"}]},
    {"tools": []},
    {"stream": True},
])
def test_optional_preflight_refuses_unsupported_shape(change):
    assert shape(**change) is None


@pytest.mark.asyncio
async def test_live_protected_text_path_uses_same_builder(monkeypatch):
    marker = {"model": "local-research-model", "max_tokens": 123,
        "messages": [{"role": "user", "content": "Selected evidence."}]}
    monkeypatch.setattr(request, "protected_text_request_arguments",
        lambda *args, **kwargs: marker, raising=False)
    protection = EvidenceProtection("Selected evidence.", 10000, 123, 0)

    async def forbidden(*args, **kwargs):
        pytest.fail("protected source preflight must not compress through a model")

    with protect_evidence(protection):
        result = await request.prepare_request(provider(), system_text="Grounding rules.",
            message_list=[{"role": "user", "content": "Selected evidence."}],
            max_tokens=123, thinking_budget=0, tool_choice_raw=None, kwargs={},
            compress_messages=forbidden)
    assert result.arguments is marker


def test_memory_prefix_and_remote_role_remain_in_the_shaped_request(monkeypatch):
    monkeypatch.delenv("OPENAI_SYSTEM_ROLE", raising=False)
    remote = SimpleNamespace(client=SimpleNamespace(base_url="https://example.invalid/v1/"),
        base_args={"model": "research-model"})
    arguments = request.protected_text_request_arguments(remote,
        system_text="[Historical conversation context]\nEarlier context.",
        message_list=[{"role": "user", "content": "Selected evidence."}], max_tokens=123)
    system = arguments["messages"][0]
    assert system["role"] == "developer"
    assert system["content"].startswith("You have access to retrieved user memory")
    assert "a tool is available or whether storage will happen behind the scenes.\n\n" in system["content"]
    assert system["content"].endswith("[Historical conversation context]\nEarlier context.")
    assert "extra_body" not in arguments


@pytest.mark.asyncio
async def test_unsupported_optional_media_keeps_the_existing_live_media_path():
    parts = [{"type": "text", "text": "Selected evidence."},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    protection = EvidenceProtection("Selected evidence.", 100000, 123, 0)

    async def forbidden(*args, **kwargs):
        pytest.fail("protected media must not invoke model compression")

    with protect_evidence(protection):
        prepared = await request.prepare_request(provider(), system_text="Grounding rules.",
            message_list=[{"role": "user", "content": parts}], max_tokens=123,
            thinking_budget=0, tool_choice_raw=None, kwargs={}, compress_messages=forbidden)
    assert prepared.arguments["messages"][-1]["content"] == parts
    assert prepared.arguments["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
