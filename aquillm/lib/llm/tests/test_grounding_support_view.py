"""Prompt coverage separates unassessed requests from selected-source gaps."""

import json

from apps.chat.services.rag_coverage import CoverageAssessment
from apps.chat.services.rag_support_delivery import (
    refresh_answer_support,
    support_grounding,
)
from apps.chat.tests.test_rag_answer_support import packet_for


def _view(packet):
    prompt = support_grounding(packet)
    return json.loads(
        prompt.split("Bounded answer_support: ", 1)[1].splitlines()[0]
    ), prompt


def test_unassessed_planner_gaps_are_not_reported_as_missing_evidence():
    packet = packet_for(
        "The selected study reports a benefit under dry conditions only.",
        "Compare the benefits and conditions.",
    )
    packet.coverage_assessment = CoverageAssessment(
        requested_aspects=("benefits", "conditions"),
        unresolved_aspects=("benefits", "conditions"),
    )
    support = refresh_answer_support(packet)
    view, prompt = _view(packet)
    assert view["state"] == "unknown"
    assert view["assessment_status"] == "unassessed"
    assert view["unassessed_aspects"] == ["benefits", "conditions"]
    assert view["unresolved_aspects"] == []
    assert view["selected_evidence_exhaustive"] is False
    assert "earlier, unvalidated search hints" in prompt
    assert "newly delivered evidence" in prompt
    # The model-facing view must not promote the conservative support state.
    assert packet.answer_support == support
    assert support.unresolved_aspects == ("benefits", "conditions")
    assert support.assessment_basis == "unvalidated"
    assert (
        packet.chunks[0]["text"]
        == "The selected study reports a benefit under dry conditions only."
    )


def test_deterministic_lookup_retains_a_scoped_unresolved_field():
    packet = packet_for(
        "The capacity is 18 litres.", "What is the capacity and what is the mass?"
    )
    view, _ = _view(packet)
    assert view["assessment_status"] == "assessed_selected_sources"
    assert view["supported_aspects"] == ["capacity"]
    assert view["unresolved_aspects"] == ["mass"]
    assert view["unassessed_aspects"] == []
    assert view["selected_evidence_exhaustive"] is False
