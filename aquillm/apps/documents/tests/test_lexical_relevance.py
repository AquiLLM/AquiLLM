"""Real candidate collection must rank whole scientific terms before LIMIT."""

import pytest
from django.test import override_settings

from apps.documents.models import DocumentRepresentationState, TextChunk
from apps.documents.services.chunk_search_candidates import (
    collect_hybrid_candidate_snapshot,
)
from .lexical_owner_support import unchunked as legacy
from .representation_support import reference

pytestmark = pytest.mark.django_db(transaction=True)


def replace_chunks(legacy, bodies):
    """Build initial exact slices only; never replace an existing representation."""
    doc, _ = legacy
    assert not TextChunk.objects.filter(doc_id=doc.id).exists()
    assert not DocumentRepresentationState.objects.filter(document_id=doc.id).exists()
    text = "\n".join(bodies)
    doc.full_text = text
    doc.save(dont_rechunk=True)
    legacy[1] = reference(doc)
    start, rows = 0, []
    for number, body in enumerate(bodies):
        rows.append(
            TextChunk(
                doc_id=doc.id,
                content=body,
                chunk_number=number,
                start_position=start,
                end_position=start + len(body),
            )
        )
        start += len(body) + 1
    return TextChunk.objects.bulk_create(rows)


def collect(legacy, query):
    return collect_hybrid_candidate_snapshot(
        TextChunk, query, 10, (legacy[0],), query_embedding=None
    )


@override_settings(RAG_CANDIDATE_MULTIPLIER=1, RAG_TRIGRAM_SIMILARITY_MIN=1)
def test_twelve_early_partial_hits_cannot_evict_later_complete_answer(legacy):
    rows = replace_chunks(legacy, ["ATP distractor"] * 12 + ["ATP binds BRCA1"])
    result = collect(legacy, "ATP BRCA1")
    assert result.exact_chunk_ids[0] == rows[-1].pk
    assert len(result.exact_chunk_ids) == 12


@pytest.mark.parametrize(
    "query,good,bad",
    [
        ("CO", "CO binds", "cooperate co CO2"),
        ("NO", "NO signal", "no signal NOTHING"),
        ("pH 7", "at pH 7 today", "at ph 7 or pH 70"),
        ("S0-2", "star S0-2 moves", "S0-20 and s0-2"),
        ("C_model", "C_model agrees", "c_model C_models"),
        ("α", "α response", "αβ response"),
        ('"binding energy"', "Binding energy measured", "unbinding energy measured"),
    ],
)
def test_scientific_exact_boundaries_and_case(legacy, query, good, bad):
    rows = replace_chunks(legacy, [bad, good])
    assert collect(legacy, query).exact_chunk_ids == (rows[1].pk,)


def test_exact_extraction_has_finite_query_and_term_bounds():
    from apps.documents.services.lexical_candidates import extract_exact_terms

    terms = extract_exact_terms(" ".join(f"ID{i}" for i in range(5000)))
    assert len(terms) <= 8
    assert all(len(term.text) <= 96 for term in terms)


def test_partial_or_match_cannot_request_protection(legacy):
    from apps.documents.services.lexical_candidates import strong_exact_ids

    rows = replace_chunks(legacy, ["ATP alone", "BRCA1 alone", "ATP BRCA1"])
    assert strong_exact_ids(rows, "ATP BRCA1") == (rows[2].pk,)


def test_explicit_symbol_question_finds_the_definition_before_examples(legacy):
    rows = replace_chunks(legacy, [
        "T-channel spectroscopy measurements.",
        "Thermometry (T) is discussed elsewhere.",
        "Viscosity (D) is the transport coefficient.",
        "Viscosity (t) describes a different quantity.",
        "Viscosity (T) was used in the T-channel spectroscopy experiment.",
        "Viscosity (T), a measure of the transport rate in this model.",
    ])
    result = collect(legacy, "Why use T-channel spectroscopy and why is viscosity also denoted by T?")
    assert result.exact_chunk_ids[0] == rows[-1].pk
    assert rows[3].pk not in result.exact_chunk_ids


@pytest.mark.parametrize("symbol", ["T1", "T-channel", "T_count", "T/R", "T^2", "T:R"])
def test_notation_matching_does_not_shorten_an_identifier(symbol):
    from apps.documents.services.lexical_candidates import extract_exact_terms

    terms = extract_exact_terms(f"The rate is represented by {symbol}.")
    assert not any(term.kind == "notation" for term in terms)


@pytest.mark.parametrize(
    "ending, expected", [("a thermal scale", []), ("'a'", ["energy a"]), ("the letter a", ["energy a"])]
)
def test_notation_distinguishes_an_article_from_a_named_symbol(ending, expected):
    from apps.documents.services.lexical_candidates import extract_exact_terms

    terms = extract_exact_terms(f"Is energy represented by {ending}?")
    assert [term.text for term in terms if term.kind == "notation"] == expected


def test_calculation_question_finds_components_under_the_printed_method_name(legacy):
    rows = replace_chunks(legacy, [
        "air_flow_vel velocities are tabulated here.",
        "An AirFlow observation was excluded from the sample.",
        "Both velocity components are obtained with the AirFlow procedure.",
    ])
    query = "How are air_flow_vel velocities calculated?"
    assert collect(legacy, query).exact_chunk_ids[0] == rows[2].pk


@pytest.mark.parametrize("query", [
    "How are air_flow_vel pressures calculated?",
    "List air_flow_vel velocities.",
    "How are x_y_vel velocities calculated?",
])
def test_compact_method_alias_requires_a_method_and_matching_quantity(query):
    from apps.documents.services.lexical_candidates import extract_exact_terms

    assert not any(term.kind == "compact_method" for term in extract_exact_terms(query))


def test_method_expansion_keeps_direct_field_answer_and_protection(legacy):
    from apps.documents.services.lexical_candidates import strong_exact_ids

    rows = replace_chunks(legacy, ["AirFlow measured humidity."] * 15 + [
        "For air_flow_vel velocities, divide displacement by time.",
    ])
    query = "How do I compute air_flow_vel velocities?"
    assert rows[-1].pk in collect(legacy, query).exact_chunk_ids[:2]
    assert strong_exact_ids(rows, query) == (rows[-1].pk,)


def test_method_verb_in_a_different_sentence_is_not_an_explanation(legacy):
    rows = replace_chunks(legacy, [
        "AirFlow. We measured velocities.",
        "The AirFlow component measures velocities.",
    ])
    query = "How do I compute air_flow_vel velocities?"
    assert collect(legacy, query).exact_chunk_ids[0] == rows[-1].pk


@pytest.mark.parametrize("question", ["What is represented by A?", "Why is this represented by T?"])
def test_an_unnamed_quantity_is_not_an_explicit_notation_relation(question):
    from apps.documents.services.lexical_candidates import extract_exact_terms

    assert not any(term.kind == "notation" for term in extract_exact_terms(question))


def test_quoted_single_letter_keeps_its_original_caseless_phrase_semantics():
    from apps.documents.services.lexical_candidates import extract_exact_terms

    terms = extract_exact_terms('Compare "Ω"')
    assert [(term.text, term.kind, term.case_sensitive) for term in terms] == [("Ω", "phrase", False)]
