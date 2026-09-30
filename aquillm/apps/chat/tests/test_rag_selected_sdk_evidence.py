"""Selected final evidence survives the real OpenAI request context path."""

from types import SimpleNamespace

import pytest

from apps.chat.services.rag_evidence_handoff import prepare_evidence_handoff
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.chat.tests.rag_evidence_handoff_support import _fixture
from lib.llm.providers.openai import OpenAIInterface


class Dispatched(Exception):
    pass


def _selected_request(monkeypatch, context_limit):
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", str(context_limit))
    monkeypatch.setenv("OPENAI_CONTEXT_RESERVE_MODE", "fixed")
    monkeypatch.setenv("OPENAI_CONTEXT_GUARD_TOKENS", "64")
    monkeypatch.setenv("OPENAI_ESTIMATOR_PAD_TOKENS", "0")
    monkeypatch.setenv("OPENAI_COMPAT_PROMPT_SLACK_TOKENS", "0")
    monkeypatch.setenv("CONTEXT_PACKER_ENABLED", "0")
    monkeypatch.setenv("TOKEN_EFFICIENCY_ENABLED", "0")
    monkeypatch.setenv("OPENAI_STREAM_RESPONSES", "0")
    convo, packet = _fixture()
    convo.messages[0].content = "old history " * 5000
    row = packet.chunks[0]
    row["text"] = "x" * 1200 + " threshold=7.3 ug/L"
    packet.chunks = [row]
    packet.citation_tokens = [row["citation"]]
    packet.image_urls = []
    request = prepare_evidence_handoff(convo, packet)[1]
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        raise Dispatched

    provider = OpenAIInterface(
        SimpleNamespace(
            base_url="http://vllm:8000",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
        "test",
    )
    return (
        convo,
        packet,
        request[-1].content,
        len(request.messages) + 1,
        calls,
        provider,
    )


@pytest.mark.asyncio
async def test_preflight_trims_history_but_sdk_receives_exact_selected_evidence(
    monkeypatch,
):
    convo, packet, expected, original_count, calls, provider = _selected_request(
        monkeypatch, 16000
    )
    with pytest.raises(Dispatched):
        await synthesize_from_evidence(provider, convo, packet, max_tokens=256)
    assert len(calls) == 1
    messages = calls[0]["messages"]
    assert len(messages) < original_count
    assert messages[-1]["content"].endswith("\n" + expected)
    assert "threshold=7.3 ug/L" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_preflight_cannot_dispatch_when_selected_evidence_will_not_fit(
    monkeypatch,
):
    convo, packet, _, _, calls, provider = _selected_request(monkeypatch, 500)
    result = await synthesize_from_evidence(provider, convo, packet, max_tokens=256)
    assert calls == []
    assert "could not deliver the selected evidence" in result[-1].content.lower()


@pytest.mark.asyncio
async def test_provider_overflow_cannot_retry_with_a_clipped_selected_message(
    monkeypatch,
):
    convo, packet, expected, _, _, provider = _selected_request(monkeypatch, 16000)
    calls = []

    async def overflow(**kwargs):
        calls.append(kwargs)
        raise ValueError("maximum context length is 1000 tokens; requested 2000 tokens")

    provider.client.chat.completions.create = overflow
    result = await synthesize_from_evidence(provider, convo, packet, max_tokens=256)
    assert len(calls) == 1
    assert calls[0]["messages"][-1]["content"].endswith("\n" + expected)
    assert "could not deliver the selected evidence" in result[-1].content.lower()
