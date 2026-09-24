"""Protected selected tool evidence reaches actual provider SDK arguments."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

TEXT = (
    '{"result":[{"text":"Policy exception: do not retain. Version 4.2: --strict. '
    'Result: 42 mK.","citation":"[doc:abc chunk:1]"}]}'
)


@pytest.mark.asyncio
async def test_openai_preparation_bypasses_all_mutating_transforms(monkeypatch):
    from lib.llm.evidence_guard import EvidenceProtection, protect_evidence
    from lib.llm.providers.openai_request import prepare_request
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    estimate = Mock(return_value=9000)
    provider = SimpleNamespace(
        client=SimpleNamespace(base_url="http://vllm:8000"),
        base_args={"model": "test"},
        _estimate_prompt_tokens=estimate,
        _preflight_trim_for_context=lambda *args: pytest.fail(
            "protected evidence trimmed"
        ),
    )
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")

    def compress(*_):
        pytest.fail("protected evidence compressed")

    budget = TurnBudget(TurnLimits())
    with protect_evidence(
        EvidenceProtection(TEXT, 16000, 512, 800, turn_budget=budget)
    ):
        request = await prepare_request(
            provider,
            system_text="grounding",
            message_list=[{"role": "user", "content": TEXT}],
            max_tokens=512,
            thinking_budget=0,
            tool_choice_raw=None,
            kwargs={},
            compress_messages=compress,
        )
    assert request.arguments["messages"][1]["content"] == TEXT
    estimate.assert_not_called()
    assert budget.text_used["tokenized"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("denial", ["closed", "exhausted"])
async def test_openai_protected_denial_skips_legacy_estimate(denial, monkeypatch):
    from lib.llm.evidence_guard import (
        ContextLimited,
        EvidenceProtection,
        protect_evidence,
    )
    from lib.llm.providers.openai_request import prepare_request
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    estimate = Mock(return_value=9000)
    provider = SimpleNamespace(
        client=SimpleNamespace(base_url="http://vllm:8000"),
        base_args={"model": "test"},
        _estimate_prompt_tokens=estimate,
    )
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    budget = TurnBudget(TurnLimits(tokenized_codepoints=1))
    if denial == "closed":
        budget.close("test")
    else:
        assert budget.reserve_text(1, kind="tokenized")
    charged_before = budget.text_used["tokenized"]
    protection = EvidenceProtection(TEXT, 16000, 512, 800, turn_budget=budget)
    with protect_evidence(protection), pytest.raises(ContextLimited):
        await prepare_request(
            provider,
            system_text="grounding",
            message_list=[{"role": "user", "content": TEXT}],
            max_tokens=512,
            thinking_budget=0,
            tool_choice_raw=None,
            kwargs={},
            compress_messages=lambda *_: pytest.fail("protected evidence compressed"),
        )
    assert protection.limited_reason == "request_tokenization_limit"
    estimate.assert_not_called()
    assert budget.text_used["tokenized"] == charged_before


@pytest.mark.asyncio
async def test_openai_legacy_preparation_still_estimates_for_compression(monkeypatch):
    from lib.llm.providers.openai_request import prepare_request

    estimate = Mock(return_value=900)
    compressions = []

    provider = SimpleNamespace(
        client=SimpleNamespace(base_url="http://vllm:8000"),
        base_args={"model": "test"},
        _estimate_prompt_tokens=estimate,
        _preflight_trim_for_context=lambda *args: None,
        _env_int=lambda *_: 256,
    )
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "1000")
    messages = [{"role": "user", "content": "hello"}]
    await prepare_request(
        provider,
        system_text="grounding",
        message_list=messages,
        max_tokens=512,
        thinking_budget=0,
        tool_choice_raw=None,
        kwargs={},
        compress_messages=lambda rows: compressions.append(rows),
    )
    estimate.assert_called_once()
    assert estimate.call_args.args[0][1] is messages[0]
    assert compressions == [messages]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["claude", "gemini"])
async def test_actual_sdk_arguments_preserve_payload(provider_name, monkeypatch):
    from lib.llm.evidence_guard import EvidenceProtection, protect_evidence
    from lib.llm.types.messages import ToolMessage

    class Dispatched(Exception):
        pass

    call = AsyncMock(side_effect=Dispatched)
    tool = ToolMessage(
        tool_name="vector_search",
        for_whom="assistant",
        content=TEXT,
        result_dict={},
        arguments={},
    )
    if provider_name == "claude":
        from lib.llm.providers.claude import ClaudeInterface

        provider = ClaudeInterface(
            SimpleNamespace(messages=SimpleNamespace(create=call))
        )
    else:
        from lib.llm.providers.gemini import GeminiInterface

        provider = GeminiInterface(
            SimpleNamespace(
                aio=SimpleNamespace(models=SimpleNamespace(generate_content=call))
            )
        )
    monkeypatch.setenv("TOKEN_EFFICIENCY_ENABLED", "1")
    monkeypatch.setenv("CONTEXT_PACKER_ENABLED", "1")
    with (
        protect_evidence(EvidenceProtection(TEXT, 16000, 512, 800)),
        pytest.raises(Dispatched),
    ):
        await provider.get_message(
            system="grounding",
            messages=[tool.render(include={"role", "content"})],
            messages_pydantic=[tool],
            max_tokens=512,
        )
    args = call.call_args.kwargs
    if provider_name == "claude":
        assert TEXT in args["messages"][0]["content"]
    else:
        assert args["contents"][0].parts[0].function_response.response["output"] == TEXT


def test_guard_includes_image_tools_history_and_metadata():
    from lib.llm.evidence_guard import (
        ContextLimited,
        EvidenceProtection,
        protect_evidence,
        validate_request,
    )

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TEXT},
                    {"type": "image_url", "image_url": {"url": "data:abc"}},
                ],
            }
        ],
        "tools": [{"description": "schema " * 400}],
    }
    with (
        protect_evidence(EvidenceProtection(TEXT, 2500, 512, 800)),
        pytest.raises(ContextLimited),
    ):
        validate_request(payload, output_reserve=512)


def test_context_overflow_never_retries_clipped_evidence():
    from lib.llm.evidence_guard import (
        ContextLimited,
        EvidenceProtection,
        protect_evidence,
    )
    from lib.llm.providers.openai_overflow import retry_args_for_context_overflow

    with (
        protect_evidence(EvidenceProtection(TEXT, 16000, 512, 800)),
        pytest.raises(ContextLimited),
    ):
        retry_args_for_context_overflow(
            {"messages": [{"role": "user", "content": TEXT}], "max_tokens": 512},
            ValueError("maximum context length is 1000 tokens. requested 2000 tokens"),
        )


def test_image_row_estimate_keeps_text_cost_and_adds_image_allowance():
    from lib.llm.evidence_guard import estimate_request_tokens

    short = {"type": "image", "text": "caption", "image_url": "/aquillm/image/1"}
    long = {**short, "text": "caption " * 1000}
    assert estimate_request_tokens(long) > estimate_request_tokens(short) + 500
    assert estimate_request_tokens(short) > 4096


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("provider_name", ["openai", "gemini", "claude"])
async def test_synthesis_through_complete_turn_reaches_sdk_with_exact_tool_payload(
    provider_name, monkeypatch
):
    from apps.chat.services.rag_evidence_handoff import prepare_evidence_handoff
    from apps.chat.services.rag_synthesis import synthesize_from_evidence
    from apps.chat.tests.test_rag_source_synthesis import (
        authorized_source_scope,
        inputs,
    )
    from lib.llm.evidence_guard import current_protection
    from lib.llm.providers.request_observability import observability_scope

    class Dispatched(Exception):
        pass

    captured = []

    async def dispatch(**kwargs):
        captured.append(kwargs)
        assert current_protection() is not None
        raise Dispatched

    call = AsyncMock(side_effect=dispatch)
    if provider_name == "openai":
        from lib.llm.providers.openai import OpenAIInterface

        provider = OpenAIInterface(
            SimpleNamespace(
                base_url="http://vllm:8000",
                chat=SimpleNamespace(completions=SimpleNamespace(create=call)),
            ),
            "test",
        )
    elif provider_name == "gemini":
        from lib.llm.providers.gemini import GeminiInterface

        provider = GeminiInterface(
            SimpleNamespace(
                aio=SimpleNamespace(models=SimpleNamespace(generate_content=call))
            )
        )
    else:
        from lib.llm.providers.claude import ClaudeInterface

        provider = ClaudeInterface(
            SimpleNamespace(messages=SimpleNamespace(create=call))
        )
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    monkeypatch.setenv("CONTEXT_PACKER_ENABLED", "1")
    monkeypatch.setenv("TOKEN_EFFICIENCY_ENABLED", "1")
    convo, packet = inputs()
    expected = prepare_evidence_handoff(convo, packet)[1][-1].content
    with (
        authorized_source_scope(packet, monkeypatch),
        observability_scope("test", "direct_synthesis"),
        pytest.raises(Dispatched),
    ):
        await synthesize_from_evidence(provider, convo, packet)
    if provider_name == "gemini":
        actual = (
            captured[0]["contents"][-1].parts[0].function_response.response["output"]
        )
        assert actual == expected
    else:
        assert captured[0]["messages"][-1]["content"].endswith("\n" + expected)
    assert convo[0].content == "old evidence"
    assert current_protection() is None
