"""Factual lead-ins need local citations even when the following list is cited."""

from copy import deepcopy

import pytest

from apps.chat.services.rag_model_answer import prepare_model_answer
from lib.llm.answer_publication import FinalCitationPolicy, validate_final_answer
from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.providers.request_observability import observability_scope
from lib.llm.tests.test_answer_publication import synthetic_publication
from lib.llm.tests.test_direct_synthesis_grounding import (
    _conversation,
    _selected_packet,
)
from lib.llm.types.response import LLMResponse


@pytest.mark.parametrize("stage", ["direct_synthesis", "post_tool_synthesis"])
async def test_repair_cites_numeric_introduction_and_keeps_useful_answer(
    stage,
):
    lead = (
        "In the 2042 report, the two configurations have different measured outcomes:"
    )
    body = (
        "\n\n- Dry configuration: 18% improvement [doc:a chunk:1]."
        "\n- Alternate configuration: 7% improvement [doc:b chunk:2]."
    )
    first = lead + body
    repaired = lead[:-1] + " [doc:a chunk:1] [doc:b chunk:2]:" + body
    requests = []

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            requests.append(deepcopy(kwargs))
            return LLMResponse(
                text=first if len(requests) == 1 else repaired,
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
            )

    convo = _conversation()
    with (
        synthetic_publication(packet=_selected_packet(convo)),
        observability_scope("lead-in", stage),
    ):
        result, _ = await complete_conversation_turn(Model(), convo, 1024)
    assert len(requests) == 2
    prompt = requests[1]["messages"][-1]["content"]
    assert "Uncited numeric sentences to revise:" in prompt
    assert lead in prompt.split("Uncited numeric sentences to revise:", 1)[1]
    assert "factual introductory prose" in prompt
    assert result[-1].content == repaired
    assert result[-1]._validated_answer.outcome == "partial"


def test_scientific_brackets_do_not_waive_uncited_numeric_intro():
    convo = _conversation()
    packet = _selected_packet(convo)
    draft = (
        "In the 2042 report, the configurations differ:\n\n"
        "- The [A II]/[B III] ratio is 0.4 [doc:a chunk:1]."
    )
    assert prepare_model_answer(packet, draft) == draft
    policy = FinalCitationPolicy(
        frozenset(packet.citation_tokens), True, "evidence", "scope"
    )
    assert validate_final_answer(draft, policy).reasons == ("uncited_numeric_claim",)


def test_citation_validation_does_not_prove_semantic_support():
    # A wrong semantic claim may pass the mechanical contract. This is a scope
    # limitation, not a model critic or an entailment regression expectation.
    convo = _conversation()
    packet = _selected_packet(convo)
    unsupported = "The wet condition universally improves performance [doc:a chunk:1]."
    policy = FinalCitationPolicy(
        frozenset(packet.citation_tokens), True, "evidence", "scope"
    )
    assert validate_final_answer(unsupported, policy).valid


@pytest.mark.parametrize(
    "source,answer",
    [
        (
            "No statistically significant difference was observed "
            "under dry conditions.",
            "The study reports no statistically significant difference under dry "
            "conditions [doc:a chunk:1]. This leaves other conditions unestablished "
            "by the selected excerpt.",
        ),
        (
            "The [A II]/[B III] ratio was 0.4 under dry conditions.",
            "The reported [A II]/[B III] ratio is 0.4 under dry conditions "
            "[doc:a chunk:1]. The selected passage does not establish a comparison "
            "with wet conditions.",
        ),
    ],
)
async def test_qualified_findings_and_honest_partial_prose_keep_model_text(
    source, answer
):
    from apps.chat.tests.test_rag_answer_support import packet_for

    packet = packet_for(source, "Explain the findings and their limitations.")
    with synthetic_publication(packet=packet) as publication:
        final = await publication.finalize(answer)
        assert final.content == answer
        assert final.outcome == "partial"
        assert packet.answer_support.state == "unknown"
        assert packet.answer_support.assessment_basis == "unvalidated"
        for sink in ("stream", "refresh", "persist"):
            await publication.verify(final, content=answer, sink=sink)
