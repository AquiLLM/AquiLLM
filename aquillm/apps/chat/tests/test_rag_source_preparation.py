"""Exact evidence representations survive beyond public previews."""

import pytest

from lib.retrieval.evidence import SourceEvidence, SourceSpan, fingerprint_source


def source(text):
    return SourceEvidence(1, "doc", 0, fingerprint_source(text), text)


@pytest.mark.parametrize(
    "tail",
    [
        "The measured result was 42 ± 2 mK only below 1 Pa.",
        "Exception: do not retain records after consent is withdrawn.",
        "In version 4.2, use --strict; --legacy is not supported.",
    ],
)
def test_full_evidence_retains_qualified_tail(tail):
    from apps.chat.services.rag_source_preparation import (
        prepare_evidence,
        render_evidence,
    )

    original = source("Background. " * 150 + tail)
    prepared = prepare_evidence(
        original, question="details", windows=(), token_ceiling=3500
    )
    assert render_evidence(prepared) == original.text
    assert prepared.source_coverage == "complete"


def test_oversized_source_keeps_exact_qualified_windows_not_prefix():
    from apps.chat.services.rag_source_preparation import (
        prepare_evidence,
        render_evidence,
    )

    text = "Irrelevant background. " * 2000 + "Exception: retain only with consent."
    original = source(text)
    span = SourceSpan(
        1,
        original.source_fingerprint,
        text.index("Exception:"),
        len(text),
        text[text.index("Exception:") :],
    )
    prepared = prepare_evidence(
        original, question="consent", windows=(span,), token_ceiling=80
    )
    assert span in prepared.spans
    assert span.text in render_evidence(prepared)
    assert prepared.source_coverage == "partial"
    assert prepared.estimated_tokens <= 80


def test_unqualified_oversized_source_exposes_unknown():
    from apps.chat.services.rag_source_preparation import prepare_evidence

    prepared = prepare_evidence(
        source("irrelevant " * 5000), question="consent", windows=(), token_ceiling=50
    )
    assert prepared.spans == ()
    assert prepared.source_coverage == "unknown"


def test_multi_span_render_has_distinct_input_fingerprint():
    from apps.chat.services.rag_source_preparation import (
        prepare_evidence,
        render_evidence,
    )

    original = source("first qualification\n" + "unrelated " * 5000 + "tail exception")
    spans = tuple(
        SourceSpan(1, original.source_fingerprint, a, b, original.text[a:b])
        for a, b in ((0, 20), (len(original.text) - 14, len(original.text)))
    )
    prepared = prepare_evidence(
        original, question="exceptions", windows=spans, token_ceiling=100
    )
    assert "[... source omitted ...]" in render_evidence(prepared)
    assert fingerprint_source(render_evidence(prepared)) != original.source_fingerprint
