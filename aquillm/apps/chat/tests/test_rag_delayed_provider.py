"""Exercise actual completion repair after the retrieval deadline."""

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    "provider_name,finish",
    [
        ("openai", "stop"),
        ("openai", "length"),
        ("claude", "stop"),
        ("claude", "length"),
        ("gemini", "stop"),
    ],
)
async def test_delayed_openai_complete_preserves_frozen_evidence_through_recovery(
    provider_name, finish, monkeypatch
):
    from apps.chat.services.rag_synthesis import synthesize_from_evidence
    from apps.chat.tests.test_rag_source_synthesis import (
        authorized_source_scope,
        inputs,
    )
    from apps.documents.services.source_loading import current_source_runtime
    from lib.llm.evidence_guard import current_protection
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.providers.request_observability import observability_scope

    now, dispatched = [0.0], []
    convo, packet = inputs()
    citation = packet.chunks[0]["citation"]

    async def create(**kwargs):
        protection = current_protection()
        dispatched.append((kwargs, protection.payload))
        now[0] = 16.0
        text = (
            "Do not retain without consent"
            if len(dispatched) == 1
            else f"Do not retain without consent {citation}."
        )
        if provider_name == "claude":
            return SimpleNamespace(
                content=[SimpleNamespace(text=text)],
                stop_reason="max_tokens"
                if finish == "length" and len(dispatched) == 1
                else "end_turn",
                usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            )
        if provider_name == "gemini":
            return SimpleNamespace(function_calls=[], text=text, usage_metadata=None)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text, tool_calls=None),
                    finish_reason=finish if len(dispatched) == 1 else "stop",
                )
            ],
            usage=None,
        )

    provider = OpenAIInterface(
        SimpleNamespace(
            base_url="http://vllm:8000",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
        "test",
    )
    if provider_name == "claude":
        from lib.llm.providers.claude import ClaudeInterface

        provider = ClaudeInterface(
            SimpleNamespace(messages=SimpleNamespace(create=create))
        )
    if provider_name == "gemini":
        from lib.llm.providers.gemini import GeminiInterface

        provider = GeminiInterface(
            SimpleNamespace(
                aio=SimpleNamespace(models=SimpleNamespace(generate_content=create))
            )
        )
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    monkeypatch.setenv("LLM_CITATION_SOURCES_APPEND", "0")
    with (
        authorized_source_scope(packet, monkeypatch),
        observability_scope("test", "direct_synthesis"),
    ):
        budget = current_source_runtime().budget
        budget._clock = lambda: now[0]
        budget._deadline = 15.0
        result = await synthesize_from_evidence(provider, convo, packet)
        assert budget._synthesis.calls == len(dispatched)
        assert not budget.can_publish()
        assert budget.text_used["tokenized"] > 0
    assert len(dispatched) >= 2
    assert citation in result[-1].content
    assert "could not deliver" not in result[-1].content
    assert len({payload for _, payload in dispatched}) == 1


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("denial", ["cancel", "tokens"])
async def test_expiry_then_cancellation_or_token_limit_stops_actual_repair(
    denial, monkeypatch
):
    import asyncio

    from apps.chat.services.rag_synthesis import synthesize_from_evidence
    from apps.chat.tests.test_rag_source_synthesis import (
        authorized_source_scope,
        inputs,
    )
    from apps.documents.services.source_loading import current_source_runtime
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.providers.request_observability import observability_scope

    calls, output, now = [], [], [0.0]
    convo, packet = inputs()

    async def create(**kwargs):
        calls.append(kwargs)
        now[0] = 16
        if denial == "cancel":
            budget.close("cancelled")
        else:
            budget._text["tokenized"] = budget.limits.tokenized_codepoints
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Do not retain without consent", tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    async def stream(payload):
        output.append(payload)

    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    monkeypatch.setenv("OPENAI_STREAM_RESPONSES", "0")
    provider = OpenAIInterface(
        SimpleNamespace(
            base_url="http://vllm:8000",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
        "test",
    )
    with (
        authorized_source_scope(packet, monkeypatch),
        observability_scope("test", "direct_synthesis"),
    ):
        budget = current_source_runtime().budget
        budget._clock, budget._deadline = lambda: now[0], 15
        if denial == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await synthesize_from_evidence(
                    provider, convo, packet, stream_func=stream
                )
        else:
            result = await synthesize_from_evidence(provider, convo, packet)
            assert "limits" in result[-1].content
        assert len(calls) == 1
        assert not budget.can_publish()
    assert output == []
