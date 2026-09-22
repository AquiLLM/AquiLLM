"""Selection telemetry accepts only bounded aggregate fields."""

from apps.chat.services import rag_metrics


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
        profile_version="adaptive-evidence-v1",
        fixed_fallback_reason="incomplete_scores",
    )
    event, fields = events[0]
    assert event == "rag_direct_turn"
    assert fields["selection_mode"] == "adaptive"
    assert fields["candidate_count"] == 45
    assert fields["profile_version"] == "adaptive-evidence-v1"
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
        profile_version="document-123/secret",
        fixed_fallback_reason="doc-123",
    )
    fields = events[0][1]
    assert "selection_mode" not in fields
    assert "candidate_count" not in fields
    assert "profile_version" not in fields
    assert "fixed_fallback_reason" not in fields
