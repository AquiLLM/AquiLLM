"""Support accepted only by an independently checkable exact lookup rule."""

from dataclasses import asdict

from lib.llm.answer_support import AnswerSupport
from lib.retrieval.evidence import (
    PreparedEvidence,
    SourceEvidence,
    SourceSpan,
    fingerprint_prepared_evidence,
    fingerprint_source,
)

from .rag_answer_context import _evidence_digest, _scope_digest
from .rag_lookup_coverage import lookup_coverage, lookup_fields


def delivered_sources(packet):
    """Match final rows to revision-bound delivery; A1 remains permission authority."""
    from lib.llm.providers.rag_citations import _chunk_citation_from_row

    from .rag_source_preparation import render_evidence

    if packet.source_authorization is None:
        return ()
    prepared = {p.source.chunk_id: p for p in packet.source_evidence}
    selected = (
        {c.chunk_id: c for c in packet.selection.candidates} if packet.selection else {}
    )
    result = []
    for row in packet.chunks:
        pk, doc = row.get("chunk_id", row.get("i")), row.get("doc_id", row.get("d"))
        text = row.get("text", row.get("x", ""))
        if not text or _chunk_citation_from_row(row) not in packet.citation_tokens:
            continue
        candidate = selected.get(pk)
        if packet.selection is not None and (
            candidate is None or candidate.doc_id != doc or candidate.text != text
        ):
            continue
        evidence = prepared.get(pk)
        if evidence is not None:
            if evidence.source.document_id != doc or render_evidence(evidence) != text:
                continue
            if candidate is not None and (
                candidate.source_fingerprint != evidence.source.source_fingerprint
            ):
                continue
        else:
            candidate = selected.get(pk)
            # Clipped text cannot establish the omitted source conditions.
            if candidate is None or candidate.doc_id != doc or candidate.text != text:
                continue
            if fingerprint_source(text) != candidate.source_fingerprint:
                continue
            source = SourceEvidence(
                pk, doc, candidate.chunk_number, candidate.source_fingerprint, text
            )
            spans = (SourceSpan(pk, source.source_fingerprint, 0, len(text), text),)
            evidence = PreparedEvidence(
                source,
                spans,
                fingerprint_prepared_evidence(source, spans),
                0,
                "complete",
            )
        result.append(evidence)
    return tuple(result)


def decide_answer_support(question, packet, *, assessment, calibration_id=None):
    """Recompute from delivery. A caller-supplied policy ID is never authority.

    Semantic activation stays closed until the A3 validated registry and separate
    operator activation are integrated. Graph readiness, planner certainty and
    numerical relevance never unlock it.
    """
    from .rag_coverage import CoverageAssessment, recheck_support, validate_assessment

    if assessment == CoverageAssessment():
        assessment = None

    requested = lookup_fields(question)
    if not requested and assessment is not None:
        requested = tuple(
            a
            for a in assessment.requested_aspects[:16]
            if isinstance(a, str) and 0 < len(a) <= 200
        )
    supported = ()
    basis = "unvalidated"
    state = "unknown" if packet.chunks else "insufficient"
    if packet.selection is not None and not packet.selection.candidates:
        state = "insufficient"
    delivered = delivered_sources(packet)
    valid = (
        question == packet.query
        and bool(delivered)
        and len(delivered) == len(packet.chunks)
        and all(p.source_coverage == "complete" for p in delivered)
        and packet.retrieval_status not in {"context_limited", "no_results"}
        and not packet.auxiliary_figures
    )
    proof = (
        lookup_coverage(question, [p.source for p in delivered], None)
        if valid
        else None
    )
    if assessment is not None:
        # Reproduce the whole proof; a planner cannot label itself deterministic,
        # invent aspects, or reuse stale references.
        if proof is None or assessment != proof:
            proof = None
    if packet.retrieval_status == "partial" and (
        assessment is None or proof is None or not proof.unresolved_aspects
    ):
        proof = None
    if proof is not None:
        checked = validate_assessment(
            {
                "requested_aspects": list(proof.requested_aspects),
                "support": [asdict(ref) for ref in proof.support],
                "unresolved_aspects": list(proof.unresolved_aspects),
            },
            [p.source for p in delivered],
            {p.source.document_id for p in delivered},
        )
        checked = recheck_support(checked, delivered)
        requested = checked.requested_aspects
        supported = tuple(
            a
            for a in requested
            if a in {r.aspect for r in checked.support}
            and a not in checked.unresolved_aspects
        )
        basis = "deterministic"
        state = (
            "complete"
            if supported == requested
            else "partial"
            if supported
            else "insufficient"
        )
    return AnswerSupport(
        state,
        requested,
        supported,
        tuple(a for a in requested if a not in supported),
        _evidence_digest(packet),
        _scope_digest(packet),
        basis,
    )
