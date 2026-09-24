"""Freeze full source text or exact qualified spans before final comparison."""

from __future__ import annotations

from itertools import islice

from apps.documents.services.source_deadline import check_source_deadline
from lib.llm.utils.evidence_tokens import estimate_text_tokens
from lib.retrieval.evidence import (
    PreparedEvidence,
    SourceSpan,
    fingerprint_prepared_evidence,
)

OMISSION = "\n[... source omitted ...]\n"


def render_evidence(prepared: PreparedEvidence) -> str:
    """Omission markers are structure, never claimed as source quotations."""
    return OMISSION.join(span.text for span in prepared.spans)


def prepare_evidence(source, *, question, windows, token_ceiling, turn_budget=None):
    """Prefer the whole source; otherwise admit only supplied qualified windows.

    Windows must be exact SourceSpans in priority order (scored acquisition or
    explicit requested support). Without such support oversized coverage is
    unknown. We never infer a supporting prefix or paraphrase qualifications.
    """

    def count(text):
        check_source_deadline()
        if turn_budget is not None and text:
            if not turn_budget.reserve_text(len(text), kind="tokenized"):
                return token_ceiling + 1
            if not turn_budget.can_publish():
                return token_ceiling + 1
        return estimate_text_tokens(text)

    full_cost = count(source.text) if token_ceiling > 0 else 1
    if full_cost <= token_ceiling:
        spans = (
            (
                SourceSpan(
                    source.chunk_id,
                    source.source_fingerprint,
                    0,
                    len(source.text),
                    source.text,
                ),
            )
            if source.text
            else ()
        )
        return PreparedEvidence(
            source,
            spans,
            fingerprint_prepared_evidence(source, spans),
            full_cost,
            "complete",
        )
    chosen = []
    chosen_cost = 0
    for span in islice(windows, 45):
        check_source_deadline()
        if (
            not isinstance(span, SourceSpan)
            or span.chunk_id != source.chunk_id
            or span.source_fingerprint != source.source_fingerprint
        ):
            raise ValueError("qualified window identity differs from source")
        if (
            span.end > len(source.text)
            or source.text[span.start : span.end] != span.text
        ):
            raise ValueError("qualified window is not an exact source slice")
        if any(span.start < old.end and old.start < span.end for old in chosen):
            continue
        proposed = sorted([*chosen, span], key=lambda item: item.start)
        cost = count(OMISSION.join(item.text for item in proposed))
        if cost <= token_ceiling:
            chosen = proposed
            chosen_cost = cost
    spans = tuple(chosen)
    return PreparedEvidence(
        source,
        spans,
        fingerprint_prepared_evidence(source, spans),
        chosen_cost,
        "partial" if spans else "unknown",
    )
