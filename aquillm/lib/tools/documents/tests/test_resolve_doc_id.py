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


def test_resolve_no_match_falls_back_to_parse_error():
    u, err = resolve_doc_id_with_candidates("eb799f0b-8b2c-425f-a187-3736846", [])
    assert u is None
    assert "Invalid document ID" in err
