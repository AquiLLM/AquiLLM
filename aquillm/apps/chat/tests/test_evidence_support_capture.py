"""Sink capture requires one canonical final identity and actual saved bytes."""

import pytest


@pytest.mark.parametrize("status", ["model", "rank_fallback", "partial_model_ranked"])
def test_live_trace_does_not_count_partial_model_order_as_full_model(
    monkeypatch, status
):
    from types import SimpleNamespace

    from apps.chat.evals.evidence_quality_trace import Trace
    from apps.chat.tests.test_rag_answer_publication import setup_publication

    _, _, packet, _, _ = setup_publication()
    packet.selection = SimpleNamespace(score_status=status, candidates=())
    runtime = SimpleNamespace(observation={"delivered_packet": packet})
    monkeypatch.setattr(
        "apps.documents.services.source_loading.current_source_runtime",
        lambda: runtime,
    )
    trace = Trace({})
    trace.sink("sdk_start", {"payload": {}, "provider": "fake", "kind": "answer"})
    assert trace.rank_fallback is (status != "model")
    assert trace.result("")["selection_score_status"] == status


def test_refresh_snapshots_do_not_duplicate_stream_text():
    from apps.chat.evals.evidence_support_capture import PublicationCapture

    capture = PublicationCapture()
    capture.outbound(
        {
            "stream": {
                "message_uuid": "m",
                "role": "assistant",
                "content": "answer",
                "done": True,
            }
        }
    )
    delta = {
        "delta": {
            "messages": [
                {"message_uuid": "m", "role": "assistant", "content": "answer"}
            ]
        }
    }
    capture.outbound(delta)
    capture.outbound(delta)
    result = capture.result("m", "answer")
    assert result["publication_capture_complete"] is True
    assert (
        result["final_stream"]
        == result["final_delta"]
        == result["persisted_answer"]
        == "answer"
    )


@pytest.mark.parametrize(
    "failure",
    ["missing", "partial", "stale_id", "changed_stream", "changed_delta", "absent_db"],
)
def test_unobserved_or_mutated_publication_cannot_pass(failure):
    from apps.chat.evals.evidence_support_capture import PublicationCapture
    from apps.chat.evals.evidence_support_metrics import publication_matches

    capture = PublicationCapture()
    stream = {
        "message_uuid": "m",
        "role": "assistant",
        "content": "answer",
        "done": True,
    }
    if failure == "partial":
        stream["done"] = False
    if failure == "changed_stream":
        stream["content"] = "changed"
    if failure != "missing":
        capture.outbound({"stream": stream})
    capture.outbound(
        {
            "delta": {
                "messages": [
                    {
                        "message_uuid": "m",
                        "role": "assistant",
                        "content": "changed"
                        if failure == "changed_delta"
                        else "answer",
                    }
                ]
            }
        }
    )
    result = capture.result(
        "old" if failure == "stale_id" else "m",
        None if failure == "absent_db" else "answer",
    )
    assert not publication_matches({"answer": "answer", **result})


@pytest.mark.parametrize(
    "change", ["invalidate", "unsent", "retry", "mutate", "stale_support"]
)
def test_final_support_tracks_last_sdk_packet_after_context_exit(monkeypatch, change):
    from dataclasses import replace
    from types import SimpleNamespace

    from apps.chat.evals.evidence_quality_trace import Trace
    from apps.chat.evals.evidence_support_capture import support_delivery_matches
    from apps.chat.services.rag_support_delivery import (
        invalidate_answer_support,
        refresh_answer_support,
    )
    from apps.chat.tests.test_rag_answer_publication import setup_publication

    _, _, packet, _, _ = setup_publication()
    packet.answer_support = replace(refresh_answer_support(packet), state="complete")
    runtime = SimpleNamespace(observation={"delivered_packet": packet})
    current = [runtime]
    monkeypatch.setattr(
        "apps.documents.services.source_loading.current_source_runtime",
        lambda: current[0],
    )
    trace = Trace({})
    dispatch = {"payload": {}, "provider": "fake", "kind": "answer"}
    trace.sink("sdk_start", dispatch)
    if change == "invalidate":
        invalidate_answer_support(packet)
    elif change in ("unsent", "retry"):
        _, _, newer, _, _ = setup_publication()
        newer.query = "new question"
        refresh_answer_support(newer)
        runtime.observation["delivered_packet"] = newer
        if change == "retry":
            trace.sink("sdk_start", dispatch)
    else:
        packet.chunks[0]["text"] = "changed after SDK"
        if change == "mutate":
            refresh_answer_support(packet)
    current[0] = None
    trace.sink("turn_complete", {"at": trace.started + 1})
    row = trace.result("answer")
    assert row["support_question"] == (
        "new question" if change == "retry" else "capacity"
    )
    assert row["support_decision"]["state"] == (
        "complete" if change in ("unsent", "stale_support") else "unknown"
    )
    assert support_delivery_matches(row) is (change not in ("mutate", "stale_support"))
