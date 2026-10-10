"""Server-controlled rendering for the currently accepted support state."""

import json
import re
from dataclasses import replace

from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.providers.rag_citations import _chunk_citation_from_row

from .rag_coverage import decide_answer_support
from .rag_lookup_coverage import lookup_coverage
from .rag_support_decision import delivered_sources


def refresh_answer_support(packet):
    packet.answer_support = decide_answer_support(
        packet.query,
        packet,
        assessment=packet.coverage_assessment,
    )
    return packet.answer_support


def support_grounding(packet):
    support = decide_answer_support(
        packet.query, packet, assessment=packet.coverage_assessment
    )
    # No fingerprints, raw score metadata or policy identifiers enter prompts
    # or persisted tool results. Memory cannot fill unresolved document aspects.
    payload = {
        "state": support.state,
        "requested_aspects": support.requested_aspects,
        "supported_aspects": support.supported_aspects,
        "unresolved_aspects": support.unresolved_aspects,
        "assessment_basis": support.assessment_basis,
        "exhaustive": False,
    }
    return (
        "\n\nBounded answer_support: "
        + json.dumps(payload, ensure_ascii=True)
        + "\nAnswer only accepted supported aspects and name unresolved aspects. "
        "Unknown support permits only exact attributed excerpts with uncertainty. "
        "Insufficient evidence requires a scoped limitation, never universal absence. "
        "Profile memory is personalization, not document evidence."
    )


def invalidate_answer_support(packet):
    """A failed current authority check cannot leave an accepted support claim."""
    if packet.answer_support is None:
        return
    support = refresh_answer_support(packet)
    packet.answer_support = replace(
        support,
        state="unknown",
        supported_aspects=(),
        unresolved_aspects=support.requested_aspects,
        assessment_basis="unvalidated",
        calibration_id=None,
    )


def _safe_excerpt(text):
    # Preserve complete excerpts and conditions. Do not clip a sentence or allow
    # source markup to create links, images, nested citations or instructions.
    return (
        isinstance(text, str)
        and 0 < len(text) <= 1200
        and not re.search(r"[<>\[\]`]|!\(|[\x00-\x08\x0b-\x1f]", text)
    )


def render_supported_answer(packet):
    """Return final text/outcome, independent of unsupported provider assertions."""
    support = refresh_answer_support(packet)
    if support.state == "insufficient":
        return LIMITED_ANSWER, "limited"
    excerpts = []
    if support.assessment_basis == "deterministic":
        sources = delivered_sources(packet)
        proof = lookup_coverage(packet.query, [p.source for p in sources], None)
        if proof is not None:
            for ref in proof.support:
                if ref.aspect not in support.supported_aspects:
                    continue
                source = next(
                    p.source for p in sources if p.source.chunk_id == ref.chunk_id
                )
                excerpts.append(
                    (
                        source.text[ref.start : ref.end],
                        f"[doc:{source.document_id} chunk:{ref.chunk_id}]",
                    )
                )
        header = "The selected source states:"
        if support.unresolved_aspects:
            header += (
                "\nUnresolved in the selected evidence: "
                + ", ".join(support.unresolved_aspects)
                + "."
            )
    else:
        header = (
            "Support for an answer is uncertain in the selected evidence. "
            "These are attributed excerpts, not a conclusion about the question:"
        )
        for row in packet.chunks:
            citation = _chunk_citation_from_row(row)
            if citation in packet.citation_tokens:
                excerpts.append((row.get("text", row.get("x", "")), citation))
    points = []
    for text, citation in excerpts:
        if _safe_excerpt(text):
            points.append(
                "\n".join(
                    f"> {sentence} {citation}"
                    for line in text.splitlines()
                    for sentence in re.split(r"(?<=[.!?])\s+", line)
                )
            )
        if len(points) == 5:
            break
    if not points:
        return LIMITED_ANSWER, "limited"
    return header + "\n\n" + "\n\n".join(
        points
    ), "answer" if support.state == "complete" else "partial"
