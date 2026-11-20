"""Citation repair at the provider boundary and at direct answer finalization."""

import pytest

from lib.llm.answer_publication import validate_final_answer
from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.tests.publication_fixture_support import selected_publication
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse

ROWS = [
    {"doc_id": "trial", "chunk_id": 1, "text": "The intake log lists pump XR-11."},
    {
        "doc_id": "trial",
        "chunk_id": 2,
        "text": "Pump XR-27 ran for 8 hours only during the dry trial.",
    },
    {"doc_id": "menus", "chunk_id": 3, "text": "Lunch was available to visitors."},
]
A, B = "[doc:trial chunk:1]", "[doc:trial chunk:2]"


async def complete(draft, repair, monkeypatch, *, repair_stop=None, outcome="partial"):
    monkeypatch.setenv("RAG_APPEND_CITATION_SOURCES", "0")
    messages = [
        UserMessage(content="Which pump ran during the dry trial?"),
        ToolMessage(
            tool_name="vector_search",
            for_whom="assistant",
            content=str(ROWS),
            result_dict={"result": ROWS},
        ),
    ]
    calls, events = [], []

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            calls.append(kwargs)
            repairing = kwargs.get("_synthesis_phase") == "citation_repair"
            text = repair if repairing else draft
            return LLMResponse(
                text=text,
                tool_call={},
                stop_reason=(repair_stop if repairing else None)
                or ("stop" if text else "length"),
                input_usage=1,
                output_usage=1,
            )

    async def stream(payload):
        events.append(payload["content"])

    with selected_publication(ROWS):
        conversation, _ = await complete_conversation_turn(
            Model(),
            Conversation(system="Use evidence.", messages=messages),
            1024,
            stream_func=stream,
        )
    final = conversation[-1]
    assert events == [final.content]
    assert final._validated_answer.outcome == outcome
    return final.content, calls


@pytest.mark.parametrize("repair", ["", "- Some other unsupported claim."])
async def test_failed_repair_preserves_only_exact_complete_draft_claim(
    repair, monkeypatch
):
    draft = (
        f"- Pump XR-27 ran for 8 hours only during the dry trial {B}.\n"
        f"- Pump XR-27 ran for 8 hours in every trial {B}.\n"
        "- The pump ran continuously for a month."
    )
    final, calls = await complete(draft, repair, monkeypatch)
    assert final == f"- Pump XR-27 ran for 8 hours only during the dry trial {B}."
    assert len(calls) == 2 and "thinking_budget" not in calls[1]


async def test_repair_checks_exact_compound_identifier_in_cited_sibling(monkeypatch):
    draft = f"The dry trial used “XR-27” {A}."
    repair = f"The dry trial used “XR-27” {B}."
    final, calls = await complete(draft, repair, monkeypatch)
    assert final == repair
    assert len(calls) == 2 and "thinking_budget" not in calls[1]
    feedback = calls[1]["messages"][-1]["content"]
    assert "XR-27" in feedback and B in feedback


async def test_wrong_sibling_cannot_pass_direct_finalization_or_failed_repair(
    monkeypatch,
):
    draft = f"The dry trial used “XR-27” {A}."
    with selected_publication(ROWS) as publication:
        assert not validate_final_answer(draft, publication.policy).valid
        answer = await publication.finalize(draft)
        assert draft not in answer.content
        assert ROWS[1]["text"] in answer.content and B in answer.content
        assert "Lunch" not in answer.content
    final, calls = await complete(draft, draft, monkeypatch)
    assert draft not in final and ROWS[1]["text"] in final and B in final
    assert "Lunch" not in final
    assert len(calls) == 2


async def test_empty_repair_retains_focused_attributed_evidence_for_paraphrase(
    monkeypatch,
):
    draft = f"The dry trial used the newer pump {B}.\n- A made-up interpretation."
    final, _ = await complete(draft, "", monkeypatch)
    assert ROWS[1]["text"] in final and B in final
    assert "newer" not in final and "made-up" not in final
    assert "Lunch" not in final and "intake log" not in final


async def test_valid_paraphrase_and_derived_quantity_need_no_literal_match(monkeypatch):
    draft = f"The dry trial lasted half of 16 hours {B}."
    final, calls = await complete(draft, "", monkeypatch)
    assert final == draft and len(calls) == 1


def test_contextual_release_shorthand_does_not_require_its_title_chunk():
    rows = [
        {
            "doc_id": "trial",
            "chunk_id": 1,
            "text": "Release R8. Laboratory procedures and maintenance records.",
        },
        {
            "doc_id": "trial",
            "chunk_id": 2,
            "text": "The pump ran eight hours in the dry trial.",
        },
    ]
    with selected_publication(rows) as ctx:
        assert validate_final_answer(
            f"In R8 the pump ran eight hours {B}.", ctx.policy
        ).valid


@pytest.mark.parametrize("name", ["ORION DX4", "“Mira 2.1”"])
def test_compound_name_requires_the_row_with_that_name_and_version_together(name):
    rows = [
        {"doc_id": "trial", "chunk_id": 1, "text": "ORION DX2, OTHER DX4, Mira 2.0."},
        {
            "doc_id": "trial",
            "chunk_id": 2,
            "text": "An ORION DX4 and a Mira 2.1 ran dry.",
        },
    ]
    with selected_publication(rows) as ctx:
        assert not validate_final_answer(f"We tested {name} {A}.", ctx.policy).valid
        assert validate_final_answer(f"We tested {name} {B}.", ctx.policy).valid


async def test_focused_excerpts_cover_both_cited_documents_and_uncited_identifiers():
    def row(doc, chunk, text):
        return {
            "doc_id": doc,
            "chunk_id": chunk,
            "text": text + " Extra context." * 120,
        }

    rows = [
        row("old", 1, "Machine X7 worked well in the old dry trial."),
        row("old", 2, "Ancillary procedures were also checked."),
        row("new", 3, "Machine X7 ran in additional trials."),
        row("new", 4, "The additional tests included ORION DX4."),
    ]
    draft = (
        "Older trials used a previous design [doc:old chunk:1][doc:old chunk:2].\n"
        "Updated trials used more equipment [doc:new chunk:3][doc:new chunk:4]:\n"
        "- New equipment included ORION DX4."
    )
    with selected_publication(rows) as ctx:
        answer = await ctx.finalize(draft)
        assert "Machine X7 worked well" in answer.content
        assert "additional tests included ORION DX4" in answer.content


async def test_incomplete_repair_can_still_improve_the_recoverable_claims(monkeypatch):
    draft = f"The published trial discussed a pump {B}.\n- A made-up finding."
    repair = (
        f"- Pump XR-27 ran for 8 hours only during the dry trial {B}.\n"
        "- No citation supplied."
    )
    final, _ = await complete(draft, repair, monkeypatch)
    assert final == f"- Pump XR-27 ran for 8 hours only during the dry trial {B}."


async def test_invalid_reference_does_not_destroy_an_independent_exact_claim(
    monkeypatch,
):
    good = f"- Pump XR-27 ran for 8 hours only during the dry trial {B}."
    final, _ = await complete(
        good + "\n- It ran forever [doc:never chunk:9].", "", monkeypatch
    )
    assert final == good


def test_literal_compound_only_in_one_independent_source_requires_corroboration():
    rows = [
        *ROWS,
        {
            "doc_id": "register",
            "chunk_id": 4,
            "text": "ORION RV2Q is a separate catalog.",
        },
    ]
    draft = f"The trial uses ORION RV2Q {A}."
    with selected_publication(rows) as ctx:
        assert not validate_final_answer(draft, ctx.policy).valid
        assert validate_final_answer(
            f"The trial uses ORION RV2Q {A} [doc:register chunk:4].", ctx.policy
        ).valid
        # A bare label or number lacks an unambiguous cross-document meaning.
        assert validate_final_answer(f"The trial uses “RV2Q” {A}.", ctx.policy).valid
    rows.append(
        {"doc_id": "other", "chunk_id": 5, "text": "ORION RV2Q was a test fixture."}
    )
    with selected_publication(rows) as ctx:
        assert validate_final_answer(draft, ctx.policy).valid


def test_wrapped_claims_and_adjoining_citation_suffix_have_the_same_local_check():
    with selected_publication(ROWS) as ctx:
        for draft in (
            f"We used “XR-27”.\n{A}",
            f"- We used “XR-27”\n  during the trial {A}.",
        ):
            assert (
                "misplaced_identifier_citation"
                in validate_final_answer(draft, ctx.policy).reasons
            )
        separate = validate_final_answer(
            f"We used “XR-27”.\n\nAn intake log exists {A}.", ctx.policy
        )
        assert separate.reasons == ("uncited_numeric_claim",)


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


async def test_uncited_draft_can_focus_a_literal_excerpt_without_publishing_claims():
    rows = [
        *ROWS,
        {
            "doc_id": "register",
            "chunk_id": 4,
            "text": "ORION RV2Q was evaluated during dry trials only.",
        },
    ]
    with selected_publication(rows) as ctx:
        answer = await ctx.finalize("The most effective system was ORION RV2Q.")
    assert rows[-1]["text"] in answer.content and "most effective" not in answer.content
    assert "Lunch" not in answer.content and "intake log" not in answer.content


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
