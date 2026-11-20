"""Provider output limits on a bounded citation repair."""

import pytest

from .test_source_bound_citation_repair import A, B, complete


async def test_local_vllm_repair_reserves_output_for_visible_citation_edits(
    monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from lib.llm.providers.openai import OpenAIInterface

    monkeypatch.setenv("OPENAI_COMPAT_ENABLE_THINKING", "1")
    transport = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Cited.", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )
    )
    llm = OpenAIInterface(
        SimpleNamespace(
            base_url="http://vllm:8000/v1",
            chat=SimpleNamespace(completions=SimpleNamespace(create=transport)),
        ),
        "local",
    )
    for phase in ("initial", "citation_repair"):
        await llm.get_message(
            system="Use evidence.",
            messages=[{"role": "user", "content": "Which pump?"}],
            max_tokens=128,
            _synthesis_phase=phase,
        )
    flags = [
        call.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        for call in transport.await_args_list
    ]
    assert flags == [True, False]


@pytest.mark.parametrize("stop_reason", ["length", "max_tokens"])
async def test_cutoff_repair_cannot_displace_a_supported_original(
    monkeypatch, stop_reason
):
    good = f"- The intake log lists pump XR-11 {A}."
    draft = good + "\n- A claim without support."
    repair = f"The newer pump proved effective {B}. The unexpected consequence was"
    final, calls = await complete(draft, repair, monkeypatch, repair_stop=stop_reason)
    assert final == good and len(calls) == 2


async def test_cutoff_repair_can_recover_only_independent_complete_units(monkeypatch):
    draft = f"Trial instruments were discussed {A}.\n- Another unsupported finding."
    good = f"- Pump XR-27 ran for 8 hours only during the dry trial {B}."
    repair = good + "\nThis establishes that every other trial also"
    final, calls = await complete(draft, repair, monkeypatch, repair_stop="length")
    assert final == good and len(calls) == 2


async def test_cutoff_repair_with_unsafe_presentation_still_reaches_publication_guard(
    monkeypatch,
):
    from lib.llm.answer_publication import LIMITED_ANSWER

    good = f"- Pump XR-27 ran for 8 hours only during the dry trial {B}."
    repair = good + '\n<iframe src="https://example.org/frame"></iframe>'
    final, _ = await complete(
        "- An unverified statement.",
        repair,
        monkeypatch,
        repair_stop="length",
        outcome="limited",
    )
    assert final == LIMITED_ANSWER
