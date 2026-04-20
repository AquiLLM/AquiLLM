"""Tests for document ID parsing and prefix disambiguation."""
from __future__ import annotations

from uuid import UUID

from lib.tools.documents.ids import clean_and_parse_doc_id, resolve_doc_id_with_candidates


def test_clean_and_parse_rejects_truncated_uuid_without_candidates():
    truncated = "eb799f0b-8b2c-425f-a187-3736846"
    u, err = clean_and_parse_doc_id(truncated)
    assert u is None
    assert "Invalid document ID" in err


def test_resolve_unique_prefix_matches_truncated_form():
    full = UUID("eb799f0b-8b2c-425f-a187-373684612345")
    truncated = "eb799f0b-8b2c-425f-a187-3736846"
    u, err = resolve_doc_id_with_candidates(truncated, [full])
    assert err == ""
    assert u == full


def test_resolve_ambiguous_prefix_returns_message():
    a = UUID("00000000-0000-4000-8000-000000000001")
    b = UUID("00000000-0000-4000-8000-000000000002")
    # Shared 20-hex prefix; both documents match.
    u, err = resolve_doc_id_with_candidates("00000000-0000-4000-8000", [a, b])
    assert u is None
    assert "Ambiguous" in err


def test_resolve_no_match_with_empty_candidates_returns_no_docs_message():
    u, err = resolve_doc_id_with_candidates("eb799f0b-8b2c-425f-a187-3736846", [])
    assert u is None
    assert "No documents are available" in err


def test_resolve_valid_uuid_not_in_candidates_rejected():
    in_corpus = UUID("11111111-1111-4111-8111-111111111111")
    stranger = UUID("eb799f3b-8b2c-425f-922a-01e2d475fec1")
    u, err = resolve_doc_id_with_candidates(str(stranger), [in_corpus])
    assert u is None
    assert "not in the documents available" in err


def test_resolve_valid_uuid_in_candidates_accepted():
    u_ok = UUID("22222222-2222-4222-8222-222222222222")
    u, err = resolve_doc_id_with_candidates(str(u_ok), [u_ok])
    assert err == ""
    assert u == u_ok


def test_resolve_valid_uuid_empty_candidates_rejected():
    u_ok = UUID("33333333-3333-4333-8333-333333333333")
    u, err = resolve_doc_id_with_candidates(str(u_ok), [])
    assert u is None
    assert "No documents are available" in err
