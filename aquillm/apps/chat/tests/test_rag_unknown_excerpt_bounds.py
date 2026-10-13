"""Unknown excerpts remain substantive, exact and bounded after attribution."""

import pytest

from apps.chat.services.rag_support_delivery import render_supported_answer
from apps.chat.tests.test_rag_answer_support import packet_for
from lib.llm.answer_publication import (
    LIMITED_ANSWER,
    FinalCitationPolicy,
    validate_final_answer,
)


def render(text):
    packet = packet_for(text, "Explain the research findings.")
    answer, outcome = render_supported_answer(packet)
    assert packet.answer_support.state == "unknown"
    return packet, answer, outcome


@pytest.mark.parametrize("text", [". " * 200, " \n\t ", "…!?\n---", "_" * 80])
def test_entirely_non_alphanumeric_sources_are_limited(text):
    _, answer, outcome = render(text)
    assert (answer, outcome) == (LIMITED_ANSWER, "limited")


def test_complete_prose_span_has_one_local_citation_without_rewriting():
    text = ". " * 20 + "A finding holds.\nThe contrary finding also applies.\n"
    packet, answer, outcome = render(text)
    assert outcome == "partial"
    citation = packet.citation_tokens[0]
    assert answer.count(citation) == 1
    quoted = answer.split("\n\n", 1)[1].replace(" " + citation, "")
    assert quoted.removeprefix("> ").replace("\n> ", "\n") == text
    assert not any(
        citation in line and not any(c.isalnum() for c in line.replace(citation, ""))
        for line in answer.splitlines()
    )


def test_repeated_factual_sentences_and_contradictory_tail_are_all_retained():
    text = "Preliminary findings support the treatment. " * 40
    text += "The final analysis does not support the treatment at 5 mg."
    packet, answer, outcome = render(text)
    assert outcome == "partial"
    assert answer.count("Preliminary findings support the treatment.") == 40
    assert "The final analysis does not support the treatment at 5 mg." in answer
    assert answer.count(packet.citation_tokens[0]) <= 5
    assert len(answer) <= 6000


@pytest.mark.parametrize(
    "text",
    [
        "Capacity is 18 litres. This applies only at 20 C.",
        'The rate is 1.25 percent. "The dose is 5 mg." No benefit above 10 mg.',
        "The dose is 5 mg.\n\nOnly at 20 C.\nThe rate is 1.25 percent.",
    ],
)
def test_numeric_conditions_remain_exact_and_pass_existing_publication_validator(text):
    packet, answer, outcome = render(text)
    assert outcome == "partial"
    policy = FinalCitationPolicy(frozenset(packet.citation_tokens), True, "e", "s")
    assert validate_final_answer(answer, policy).valid
    quoted = answer.split("\n\n", 1)[1].replace(" " + packet.citation_tokens[0], "")
    assert quoted.removeprefix("> ").replace("\n> ", "\n") == text


def test_rendered_limit_rejects_complete_source_without_a_favorable_prefix():
    text = "Substantive finding.\n" * 280 + "The final condition contradicts it."
    assert len(text) < 6000
    _, answer, outcome = render(text)
    assert (answer, outcome) == (LIMITED_ANSWER, "limited")
    assert "Substantive finding" not in answer


def test_oversized_attribution_is_charged_to_rendered_limit():
    packet = packet_for("A complete finding.", "Explain the finding.")
    document = "d" * 6000
    packet.chunks[0]["doc_id"] = document
    packet.citation_tokens = [f"[doc:{document} chunk:1]"]
    assert render_supported_answer(packet) == (LIMITED_ANSWER, "limited")


def test_duplicate_whole_sources_do_not_amplify_sentence_citations():
    packet = packet_for("A finding holds. " * 30, "Explain the finding.")
    packet.chunks *= 20
    answer, outcome = render_supported_answer(packet)
    assert outcome == "partial" and packet.answer_support.state == "unknown"
    assert answer.count(packet.citation_tokens[0]) <= 5
    assert len(answer) <= 6000


@pytest.mark.parametrize("text", ["研究結果は条件付きです。", "١٢", "42"])
def test_unicode_letters_and_numeric_sources_are_not_punctuation(text):
    _, answer, outcome = render(text)
    assert outcome == "partial"
    assert text in answer
