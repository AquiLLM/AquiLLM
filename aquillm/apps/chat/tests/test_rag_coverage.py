"""Coverage gate and structural support validation."""

from types import SimpleNamespace

import pytest

from lib.retrieval.evidence import SourceEvidence, SourceSpan


@pytest.mark.parametrize(
    "question,text,needed",
    [
        ("What was the measured yield?", "The measured yield was 42 percent.", False),
        (
            "What was the measured yield?",
            "Our yield measurement method uses a balance.",
            True,
        ),
        ("Compare both.", "They agree.", True),
        (
            "What is the retention period?",
            "Retention is 30 days except pending litigation.",
            True,
        ),
        ("What was the measured yield?", "", True),
        (
            "What was the measured yield?",
            "The measured yield was 2.3 percent. The measured yield was 2.4 percent.",
            True,
        ),
    ],
)
def test_gate_uses_requested_field_and_evidence(question, text, needed):
    from apps.chat.services.rag_coverage import needs_coverage_assessment

    sources = (SourceEvidence(1, "doc", 0, "r", text),) if text else ()
    assert needs_coverage_assessment(question, sources, None) is needed


def test_retained_chunk_without_tail_support_is_still_unresolved():
    from apps.chat.services.rag_coverage import (
        CoverageAssessment,
        SupportReference,
        recheck_support,
    )

    assessment = CoverageAssessment(
        ("yield",), (SupportReference("yield", 1, "r", 90, 99),), (), None, "assessed"
    )
    delivered = (SimpleNamespace(spans=(SourceSpan(1, "r", 0, 20, "prefix"),)),)
    result = recheck_support(assessment, delivered)
    assert result.unresolved_aspects == ("yield",)
    assert not result.support


@pytest.mark.parametrize(
    "action",
    [
        {"kind": "vector", "query": "yield", "aspect": "invented"},
        {
            "kind": "document",
            "query": "yield",
            "document_id": "outside",
            "aspect": "yield",
        },
        {"kind": "adjacent", "document_id": "doc", "chunk_id": 99, "aspect": "yield"},
        {"kind": "adjacent", "document_id": "doc", "chunk_id": True, "aspect": "yield"},
    ],
)
def test_planner_action_must_name_unresolved_aspect_and_known_scope(action):
    from apps.chat.services.rag_coverage import validate_assessment

    payload = {
        "requested_aspects": ["yield"],
        "support": [],
        "unresolved_aspects": ["yield"],
        "next_action": action,
    }
    with pytest.raises(ValueError):
        validate_assessment(
            payload, (SourceEvidence(1, "doc", 0, "r", "method"),), {"doc"}
        )
