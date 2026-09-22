"""Selection telemetry accepts only bounded aggregate fields."""

from types import SimpleNamespace

from apps.chat.services import rag_metrics, rag_selection_coordinator
from apps.chat.services.rag_config import (
    EvidenceSelectionConfig,
    evidence_selection_config,
)
from apps.chat.services.rag_selection_scoring import PreparedSelection
from apps.chat.services.rag_selection_types import (
    EvidenceSelection,
    SelectionCandidate,
    SelectionProfile,
)


def _emit(monkeypatch, **selection_fields):
    events = []
    monkeypatch.setattr(
        rag_metrics.logger,
        "info",
        lambda event, **fields: events.append(fields),
    )
    rag_metrics.log_direct_rag_turn(
        intent_ms=0,
        query_ms=0,
        retrieval_ms=0,
        evidence_ms=0,
        synthesis_ms=0,
        total_ms=0,
        retrieved_count=2,
        retained_count=2,
        retrieval_status="results_found",
        **selection_fields,
    )
    return events[0]


def test_invalid_config_emits_only_fixed_closed_diagnostic(monkeypatch):
    packet = SimpleNamespace(chunks=[], total_tokens=0)
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "invalid")
    config = evidence_selection_config()
    fields = _emit(
        monkeypatch,
        **rag_selection_coordinator.selection_metric_fields(config, None, packet),
    )
    assert fields["selection_mode"] == "legacy"
    assert (
        fields["selection_config_error"] == "invalid_evidence_selection_configuration"
    )
    rejected = _emit(monkeypatch, selection_config_error="document-private-id")
    assert "selection_config_error" not in rejected


def test_shadow_proposal_is_separate_from_served_counts(monkeypatch):
    candidate = SelectionCandidate(
        7,
        "private-document-id",
        1,
        "private passage",
        0.8,
        1,
        "private-source-fingerprint",
        {},
    )
    profile = SelectionProfile("breadth", 0.8, 0.15, "adaptive-evidence-v1-minmax-1e-9")
    turn = rag_selection_coordinator.SelectionTurn(
        EvidenceSelection((candidate,), 4, profile, "rank_fallback"),
        object(),
        PreparedSelection((candidate,), "rank_fallback", 0, 0, 0, None),
        1.0,
        3,
    )
    served = SimpleNamespace(
        chunks=[{"doc_id": "served-a"}, {"doc_id": "served-b"}],
        total_tokens=50,
    )
    fields = _emit(
        monkeypatch,
        **rag_selection_coordinator.selection_metric_fields(
            EvidenceSelectionConfig(mode="shadow"),
            turn,
            served,
        ),
    )
    assert (
        fields["retained_count"],
        fields["selected_doc_count"],
        fields["estimated_tokens"],
    ) == (2, 2, 50)
    assert (
        fields["proposed_selected_count"],
        fields["proposed_selected_doc_count"],
        fields["proposed_estimated_tokens"],
        fields["proposed_profile_name"],
        fields["proposed_score_status"],
    ) == (1, 1, 4, "breadth", "rank_fallback")
    assert "private-document-id" not in repr(fields)
    assert "private passage" not in repr(fields)
    rejected = _emit(
        monkeypatch,
        proposed_profile_name="private-document-id",
        proposed_score_status="private passage",
        proposed_selected_count=46,
    )
    assert not any(key.startswith("proposed_") for key in rejected)


def test_selection_metrics_are_closed_and_bounded(monkeypatch):
    events = []
    monkeypatch.setattr(
        rag_metrics.logger,
        "info",
        lambda event, **fields: events.append((event, fields)),
    )
    rag_metrics.log_direct_rag_turn(
        intent_ms=0,
        query_ms=0,
        retrieval_ms=0,
        evidence_ms=0,
        synthesis_ms=0,
        total_ms=0,
        retrieved_count=1,
        retrieval_status="results_found",
        selection_mode="adaptive",
        selector_ms=1.2,
        final_scoring_ms=2.3,
        candidate_count=45,
        selected_doc_count=2,
        estimated_tokens=120,
        reused_pairs=3,
        new_pairs=4,
        profile_version="adaptive-evidence-v1-minmax-1e-9",
        fixed_fallback_reason="incomplete_scores",
    )
    event, fields = events[0]
    assert event == "rag_direct_turn"
    assert fields["selection_mode"] == "adaptive"
    assert fields["candidate_count"] == 45
    assert fields["profile_version"] == "adaptive-evidence-v1-minmax-1e-9"
    assert all("score" not in key for key in fields)
    events.clear()
    rag_metrics.log_direct_rag_turn(
        intent_ms=0,
        query_ms=0,
        retrieval_ms=0,
        evidence_ms=0,
        synthesis_ms=0,
        total_ms=0,
        retrieved_count=1,
        retrieval_status="results_found",
        selection_mode="raw-question",
        candidate_count=46,
        profile_version="private-document-123",
        fixed_fallback_reason="doc-123",
    )
    fields = events[0][1]
    assert "selection_mode" not in fields
    assert "candidate_count" not in fields
    assert "profile_version" not in fields
    assert "fixed_fallback_reason" not in fields
