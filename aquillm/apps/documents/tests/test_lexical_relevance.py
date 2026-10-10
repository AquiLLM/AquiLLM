"""Real candidate collection must rank whole scientific terms before LIMIT."""

from hashlib import sha256

import pytest
from django.test import override_settings

from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_candidates import (
    collect_hybrid_candidate_snapshot,
)
from apps.documents.tests.test_embedding_generation import legacy as legacy

pytestmark = pytest.mark.django_db(transaction=True)


def replace_chunks(legacy, bodies):
    doc, _ = legacy
    text = "\n".join(bodies)
    type(doc).objects.filter(pk=doc.pk).update(
        full_text=text, full_text_hash=sha256(text.encode()).hexdigest()
    )
    doc.refresh_from_db()
    TextChunk.objects.filter(doc_id=doc.id).delete()
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
