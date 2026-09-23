"""Contract tests for exact, revision-aware retrieval evidence."""

from dataclasses import FrozenInstanceError

import pytest

from lib.retrieval.evidence import (
    PreparedEvidence,
    SourceEvidence,
    SourceSpan,
    fingerprint_prepared_evidence,
    fingerprint_source,
)


def source(text="A😀β\nTail"):
    return SourceEvidence(7, "doc-2", 3, "revision-a", text)


def test_source_fingerprint_tracks_content_and_records_are_immutable():
    assert fingerprint_source("A😀β") == fingerprint_source("A😀β")
    assert fingerprint_source("A😀β") != fingerprint_source("A😀γ")
    with pytest.raises(FrozenInstanceError):
        source().text = "changed"


def test_exact_unicode_spans_validate_against_full_source():
    item = source()
    span = SourceSpan(7, "revision-a", 1, 3, "😀β")
    prepared = PreparedEvidence(
        item, (span,), fingerprint_prepared_evidence(item, (span,)), 2, "partial"
    )
    assert prepared.spans[0].text == "😀β"
    assert prepared.source_coverage == "partial"
    with pytest.raises(ValueError, match="span text"):
        PreparedEvidence(
            item, (SourceSpan(7, "revision-a", 1, 3, "😀"),), "packet-a", 2, "partial"
        )
    with pytest.raises(ValueError, match="span identity"):
        PreparedEvidence(
            item, (SourceSpan(7, "revision-b", 1, 3, "😀β"),), "packet-a", 2, "partial"
        )


def test_complete_coverage_cannot_hide_a_missing_tail():
    item = source()
    with pytest.raises(ValueError, match="source tail"):
        PreparedEvidence(
            item,
            (SourceSpan(7, "revision-a", 0, 3, "A😀β"),),
            "packet-a",
            2,
            "complete",
        )
    full = SourceSpan(7, "revision-a", 0, len(item.text), item.text)
    fingerprint = fingerprint_prepared_evidence(item, (full,))
    assert (
        PreparedEvidence(item, (full,), fingerprint, 4, "complete").source_coverage
        == "complete"
    )


@pytest.mark.parametrize("coverage", ["COMPLETE", "limited", "", None, ["complete"]])
def test_coverage_uses_only_explicit_literal_states(coverage):
    item = source()
    with pytest.raises(ValueError, match="invalid source_coverage"):
        PreparedEvidence(item, (), "packet-a", 0, coverage)


def test_prepared_evidence_rejects_out_of_order_or_overlapping_spans():
    item = source("abcdef")
    first = SourceSpan(7, "revision-a", 0, 3, "abc")
    overlap = SourceSpan(7, "revision-a", 2, 4, "cd")
    with pytest.raises(ValueError, match="overlapping"):
        PreparedEvidence(item, (first, overlap), "packet-a", 3, "partial")


def test_prepared_fingerprint_rejects_mutated_text_and_reordered_spans():
    original = source("abcdef")
    spans = (
        SourceSpan(7, "revision-a", 0, 2, "ab"),
        SourceSpan(7, "revision-a", 4, 6, "ef"),
    )
    fingerprint = fingerprint_prepared_evidence(original, spans)
    assert (
        PreparedEvidence(
            original, spans, fingerprint, 2, "partial"
        ).evidence_fingerprint
        == fingerprint
    )
    changed = source("abcdeg")
    changed_spans = (spans[0], SourceSpan(7, "revision-a", 4, 6, "eg"))
    with pytest.raises(ValueError):
        PreparedEvidence(changed, changed_spans, fingerprint, 2, "partial")
    changed_outside_spans = source("abXdef")
    with pytest.raises(ValueError):
        PreparedEvidence(changed_outside_spans, spans, fingerprint, 2, "partial")
    assert fingerprint_prepared_evidence(original, spans[::-1]) != fingerprint
