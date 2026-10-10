"""Ranking, structural provenance and answer support are different contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from apps.chat.services import rag_coverage
from apps.chat.services.rag_evidence import EvidencePacket
from apps.chat.services.rag_lookup_coverage import lookup_coverage
from lib.retrieval.evidence import (
    PreparedEvidence,
    SourceEvidence,
    SourceSpan,
    fingerprint_prepared_evidence,
    fingerprint_source,
)


def packet_for(
    text="The capacity is 18 litres at 20 C.", question="What is the capacity?"
):
    source = SourceEvidence(1, "a", 0, fingerprint_source(text), text)
    spans = (SourceSpan(1, source.source_fingerprint, 0, len(text), text),)
    prepared = PreparedEvidence(
        source, spans, fingerprint_prepared_evidence(source, spans), 12, "complete"
    )
    packet = EvidencePacket(
        [{"doc_id": "a", "chunk_id": 1, "text": text}],
        [],
        ["[doc:a chunk:1]"],
        question,
        "selected documents",
        "results_found",
        "",
        12,
        source_evidence=(prepared,),
        source_mode=True,
        source_authorization=SimpleNamespace(authorization_context_signature="scope"),
    )
    return packet


def decide(packet, assessment=None, **kwargs):
    implementation = getattr(rag_coverage, "decide_answer_support", None)
    assert callable(implementation), "answer-support state machine is missing"
    return implementation(packet.query, packet, assessment=assessment, **kwargs)


def test_rank_and_valid_citation_do_not_prove_answerability():
    packet = packet_for(
        "This excerpt describes telescope optics.", "Does this drug improve survival?"
    )
    result = decide(packet)
    assert (result.state, result.assessment_basis, result.exhaustive) == (
        "unknown",
        "unvalidated",
        False,
    )


def test_zero_authorized_delivered_sources_is_scoped_insufficient():
    packet = packet_for()
    packet.chunks = []
    packet.source_evidence = ()
    assert decide(packet).state == "insufficient"


@pytest.mark.parametrize("calibration", [None, "forged", "stale-policy-v1"])
def test_structurally_valid_planner_is_never_semantic_proof(calibration):
    packet = packet_for("Capacity is 18 litres.", "Does this drug improve survival?")
    assessment = rag_coverage.validate_assessment(
        {
            "requested_aspects": ["survival"],
            "unresolved_aspects": [],
            "support": [
                {
                    "aspect": "survival",
                    "chunk_id": 1,
                    "source_fingerprint": packet.source_evidence[
                        0
                    ].source.source_fingerprint,
                    "start": 0,
                    "end": 10,
                }
            ],
        },
        [packet.source_evidence[0].source],
        {"a"},
    )
    result = decide(packet, assessment, calibration_id=calibration)
    assert result.state == "unknown"
    assert result.calibration_id is None
    assert result.supported_aspects == ()


def test_exact_lookup_preserves_conditions_and_is_not_exhaustive():
    packet = packet_for()
    result = decide(packet)
    assert result.state == "complete"
    assert result.assessment_basis == "deterministic"
    assert result.supported_aspects == ("capacity",)
    assert not result.exhaustive


@pytest.mark.parametrize(
    "text",
    [
        "The capacity is not 18 litres.",
        'The report quotes "The capacity is 18 litres."',
        "The capacity is 18 litres except under pressure.",
        "The capacity was 18 litres in 2020; it is 12 litres now.",
        "The volume is 18 litres.",
        "The capacity estimate is 18 litres. The capacity is 12 litres.",
        "The capacity is 18 litres. The capacity is 12 litres.",
        "The capacity is 18 litres at 20 C. The capacity is 18 litres at 40 C.",
        "The capacity is 18 litres. The capacity is 18 gallons.",
        "The capacity is 18 ML.",
        "The capacity is 18 M.",
    ],
)
def test_lookup_rejects_ambiguous_scientific_conditions(text):
    packet = packet_for(text)
    assert (
        lookup_coverage(packet.query, [packet.source_evidence[0].source], None) is None
    )
    assert decide(packet).state == "unknown"


@pytest.mark.parametrize(
    "change", ["offset", "fingerprint", "invented", "question", "omitted"]
)
def test_stale_or_undelivered_deterministic_proof_cannot_complete(change):
    packet = packet_for()
    proof = lookup_coverage(packet.query, [packet.source_evidence[0].source], None)
    if change in {"offset", "fingerprint", "invented"}:
        ref = proof.support[0]
        ref = replace(
            ref,
            **{
                "offset": {"end": 10000},
                "fingerprint": {"source_fingerprint": "stale"},
                "invented": {"aspect": "survival"},
            }[change],
        )
        proof = replace(proof, support=(ref,))
    elif change == "question":
        packet.query = "Does this drug improve survival?"
    else:
        prepared = packet.source_evidence[0]
        span = replace(prepared.spans[0], end=12, text=prepared.source.text[:12])
        packet.source_evidence = (
            replace(
                prepared,
                spans=(span,),
                source_coverage="partial",
                evidence_fingerprint=fingerprint_prepared_evidence(
                    prepared.source, (span,)
                ),
            ),
        )
        packet.chunks[0]["text"] = span.text
    result = decide(packet, proof)
    assert result.state == "unknown"
    assert not result.supported_aspects


def test_missing_multipart_field_keeps_supported_portion():
    packet = packet_for(question="What is the capacity and what is the voltage?")
    result = decide(packet)
    assert result.state == "partial"
    assert result.supported_aspects == ("capacity",)
    assert result.unresolved_aspects == ("voltage",)


def test_comparison_with_missing_input_cannot_claim_consensus():
    assert (
        decide(packet_for(question="Compare capacity in papers A and B.")).state
        == "unknown"
    )


@pytest.mark.parametrize(
    "values,expected", [((-3.0, -1.0), (0.0, 1.0)), ((0.0, 0.0), (0.5, 0.5))]
)
@pytest.mark.parametrize("kind", ["pointwise", "listwise"])
def test_numeric_scores_preserve_raw_values_without_support_confidence(
    values, expected, kind
):
    from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
    from apps.chat.tests.test_rag_selection_scoring import (
        PointwiseScorer,
        Policy,
        _authorization,
        _chunk,
        _pool,
        _score_set,
    )

    chunks = (_chunk(1, "first"), _chunk(2, "second"))
    scorer = PointwiseScorer()
    scorer.scoring_kind = kind
    result = prepare_selection_candidates(
        pool=_pool(chunks, (_score_set("question", chunks, values, kind=kind),)),
        primary_query="question",
        authorization=_authorization(Policy()),
        deadline=100,
        allow_new_scores=False,
        scorer=scorer,
        chunk_loader=lambda *_: chunks,
        clock=lambda: 0,
    )
    assert tuple(c.relevance for c in result.candidates) == expected
    assert tuple(getattr(c, "raw_score", None) for c in result.candidates) == values
    assert all(c.score_interpretation == "uncalibrated" for c in result.candidates)
    assert all(c.scorer_fingerprint == "scorer-v1" for c in result.candidates)
    packet = packet_for(question="Does this drug improve survival?")
    packet.selection = SimpleNamespace(candidates=result.candidates)
    packet.source_evidence = ()
    packet.chunks = [dict(c.row) for c in result.candidates]
    packet.citation_tokens = [row["citation"] for row in packet.chunks]
    assert decide(packet).state == "unknown"


def test_rank_fallback_has_no_numerical_support_confidence():
    from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
    from apps.chat.tests.test_rag_selection_scoring import (
        Policy,
        _authorization,
        _chunk,
        _pool,
    )

    chunks = (_chunk(1, "first"),)
    result = prepare_selection_candidates(
        pool=_pool(chunks),
        primary_query="question",
        authorization=_authorization(Policy()),
        deadline=100,
        allow_new_scores=False,
        chunk_loader=lambda *_: chunks,
        clock=lambda: 0,
    )
    assert result.score_status == "rank_fallback"
    assert getattr(result.candidates[0], "score_interpretation", None) == "unavailable"
    assert result.candidates[0].raw_score is None


def test_dropped_comparison_field_is_unresolved_but_other_field_survives():
    packet = packet_for(question="What is the capacity and what is the voltage?")
    voltage = SourceEvidence(2, "a", 1, "voltage-revision", "The voltage is 12 V.")
    proof = lookup_coverage(
        packet.query, [packet.source_evidence[0].source, voltage], None
    )
    proof = rag_coverage.recheck_support(proof, packet.source_evidence)
    packet.retrieval_status = "partial"
    result = decide(packet, proof)
    assert result.state == "partial"
    assert result.supported_aspects == ("capacity",)
    assert result.unresolved_aspects == ("voltage",)


@pytest.mark.parametrize("status", ["partial", "context_limited"])
def test_budget_or_graph_omissions_cannot_become_complete(status):
    packet = packet_for()
    packet.retrieval_status = status
    assert decide(packet).state == "unknown"


def test_known_empty_authorized_selection_is_insufficient_even_with_stale_rows():
    packet = packet_for()
    packet.selection = SimpleNamespace(candidates=())
    assert decide(packet).state == "insufficient"


def test_legacy_complete_revision_can_prove_lookup_but_clipped_revision_cannot():
    from apps.chat.services.rag_selection_types import SelectionCandidate

    packet = packet_for()
    source = packet.source_evidence[0].source
    packet.source_evidence = ()
    packet.source_mode = False
    candidate = SelectionCandidate(
        1, "a", 0, source.text, 1.0, 1, source.source_fingerprint, packet.chunks[0]
    )
    packet.selection = SimpleNamespace(candidates=(candidate,))
    assert decide(packet).state == "complete"
    packet.selection = SimpleNamespace(
        candidates=(replace(candidate, source_fingerprint="full-source-revision"),)
    )
    assert decide(packet).state == "unknown"
