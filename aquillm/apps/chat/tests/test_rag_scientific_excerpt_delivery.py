"""Source notation must survive publication without becoming executable markup."""

from copy import deepcopy
from html import unescape

import pytest

from apps.chat.services.rag_answer_context import RagAnswerPublication, _evidence_digest
from apps.chat.services.rag_complete_excerpts import partition_complete_source
from apps.chat.services.rag_support_delivery import render_supported_answer
from apps.chat.tests.test_rag_answer_support import packet_for
from lib.llm.answer_publication import FinalCitationPolicy, validate_final_answer
from lib.llm.providers.rag_citations import extract_citations
from lib.llm.turn_context import TurnContext, bind_turn
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.parametrize(
    "text",
    [
        "The maximum redshift is z=4.119 [18].",
        "The sample has 0.3 < z < 1.5 and i > 22 mag [12, 18].",
        "The metrics saturate at 40-50% of GalaxiesML [15].",
        "The variable `z` denotes redshift, with uncertainty [0.01, 0.02].",
    ],
)
def test_scientific_notation_is_published_without_mutating_evidence(text):
    packet = packet_for(text, "Explain the reported scientific findings.")
    original = deepcopy(packet.chunks)
    digest = _evidence_digest(packet)
    assert partition_complete_source(text, slots=5, codepoints=6000) == (text,)

    answer, outcome = render_supported_answer(packet)

    assert outcome == "partial"
    assert text in unescape(answer.replace("<wbr>", ""))
    assert packet.chunks == original
    assert _evidence_digest(packet) == digest
    assert packet.answer_support.state == "unknown"
    policy = FinalCitationPolicy(
        frozenset(packet.citation_tokens), True, digest, "scope"
    )
    assert validate_final_answer(answer, policy).valid


def test_source_markup_and_citation_tokens_remain_literal_attributed_text():
    text = (
        '<img src="https://evil.example/pixel"> '
        "[link](https://evil.example) ![image](https://evil.example/image) "
        "`code` **bold** $x$ www.evil.example user@evil.example "
        "&lt;script&gt; [doc:foreign chunk:99]."
    )
    packet = packet_for(text, "Explain the source.")

    answer, outcome = render_supported_answer(packet)

    assert outcome == "partial"
    assert text in unescape(answer.replace("<wbr>", ""))
    assert extract_citations(answer) == ["[doc:a chunk:1]"]
    assert "<img" not in answer and "[link]" not in answer and "![image]" not in answer
    assert "www.evil.example" not in answer and "user@evil.example" not in answer
    assert validate_final_answer(
        answer, FinalCitationPolicy(frozenset(packet.citation_tokens), True, "e", "s")
    ).valid


def test_indented_sentences_keep_exact_source_and_local_citations():
    text = (
        "The preliminary finding is 1.\n\n    Conditions remain unchanged."
        "\n\tThe final estimate is 2 [18], with x < 3."
    )
    packet = packet_for(text, "Explain the findings.")
    answer, outcome = render_supported_answer(packet)
    assert outcome == "partial"
    assert validate_final_answer(
        answer, FinalCitationPolicy(frozenset(packet.citation_tokens), True, "e", "s")
    ).valid
    quote = answer.split("\n\n", 1)[1].removeprefix("> ").replace("\n> ", "\n")
    assert unescape(quote.replace(" [doc:a chunk:1]", "").replace("<wbr>", "")) == text


@pytest.mark.asyncio
async def test_publication_receipt_keeps_scientific_values_and_rejects_source_changes():
    from lib.llm.answer_publication import AnswerPublicationRejected

    packet = packet_for(
        "The maximum redshift is z=4.119 [18], with z < 5.",
        "What is the maximum redshift of GalaxiesML-Spectra?",
    )
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        original = deepcopy(packet.chunks)
        digest = _evidence_digest(packet)
        draft = "The maximum is z=4.119, with z < 5 [doc:a chunk:1]."
        answer = await context.finalize(draft)
        assert answer.outcome == "partial"
        assert answer.content == draft
        assert packet.chunks == original
        assert _evidence_digest(packet) == digest
        assert packet.answer_support.state == "unknown"
        await context.verify(answer, content=answer.content, sink="persist")
        packet.chunks[0]["text"] = "Different source text."
        with pytest.raises(AnswerPublicationRejected, match="evidence_changed"):
            await context.verify(answer, content=answer.content, sink="persist")
