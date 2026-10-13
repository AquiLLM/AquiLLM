"""Graph replay provenance distinguishes deferred baseline bodies from graph sources."""

import json
from types import SimpleNamespace
from uuid import UUID

from apps.chat.evals.replay_observations import fingerprint, summarize
from lib import replay_observation
from lib.evidence_observation import observe


def stage_event():
    rows = tuple(
        SimpleNamespace(pk=i, doc_id=UUID(int=1), content=f"private-{i}")
        for i in (101, 102, 103)
    )
    snapshot = SimpleNamespace(
        baseline_candidates=rows[:1],
        source_admission=SimpleNamespace(baseline_order=(101, 102)),
        vector_results=rows[:1],
        trigram_results=(),
        exact_results=(),
        graph_seeds=(),
        graph_seed_error=False,
        effective_limits=None,
    )
    events = []
    with observe(lambda name, data: events.append((name, data))):
        replay_observation.retrieval_stages(
            "private question",
            3,
            snapshot,
            SimpleNamespace(combined_candidates=rows, graph_candidates=()),
            rows,
            {"graph_status": "hit"},
            overlay_enabled=True,
            hybrid_pool=rows,
            graph_seed_attempted=True,
        )
    return next(data for name, data in events if name == "retrieval_stages")


def test_deferred_baseline_is_not_mislabeled_as_graph():
    graph = stage_event()["graph"]
    assert [row["chunk_id"] for row in graph["candidates"]] == [103]


def test_final_graph_membership_uses_published_ids_not_pool_novelty_or_text():
    events = [
        ("graph_selected_sources_v2", {"chunk_ids": (101, 103)}),
        ("retrieval_stages", stage_event()),
    ]
    packet = [
        {"chunk_id": 101, "text": "same"},
        {"chunk_id": 102, "text": "same"},
        {"chunk_id": 103, "text": "different excerpt"},
    ]
    result = summarize(events, packet, [])
    assert result.get("graph_source_provenance") == {
        "status": "captured",
        "selected_count": 2,
        "packet_count": 2,
    }
    assert [row["graph_selected_source"] for row in result["packet"]] == [
        True,
        False,
        True,
    ]
    assert (
        result["packet"][0]["source_identity_sha256"]
        != result["packet"][1]["source_identity_sha256"]
    )
    assert result["packet"][0]["source_identity_sha256"] == fingerprint(
        "text-chunk:101"
    )
    serialized = json.dumps(result)
    assert "chunk_id" not in serialized and "private" not in serialized


def test_no_publication_event_is_unknown_even_with_novel_pool_rows():
    result = summarize(
        [("retrieval_stages", stage_event())],
        [{"chunk_id": 103, "text": "private"}],
        [],
    )
    assert result["graph_source_provenance"] == {
        "status": "unavailable",
        "selected_count": None,
        "packet_count": None,
    }
    assert result["packet"][0]["graph_selected_source"] is None


def test_empty_published_selection_is_known_zero():
    result = summarize(
        [("graph_selected_sources_v2", {"chunk_ids": ()})],
        [{"chunk_id": 103, "text": "private"}],
        [],
    )
    assert result["graph_source_provenance"] == {
        "status": "captured",
        "selected_count": 0,
        "packet_count": 0,
    }
    assert result["packet"][0]["graph_selected_source"] is False


def test_source_identity_is_stable_across_excerpt_text_and_packet_aliases():
    rows = [
        {"chunk_id": 103, "text": "left excerpt"},
        {"id": 103, "text": "right"},
        {"i": 103, "x": "other"},
        {"chunk_id": True, "text": "unknown"},
        {"text": "missing id"},
    ]
    result = summarize([], rows, [])
    packets = result["packet"]
    assert len({row["source_identity_sha256"] for row in packets[:3]}) == 1
    assert packets[0]["text_sha256"] != packets[1]["text_sha256"]
    assert packets[3]["source_identity_sha256"] is None
    assert packets[4]["source_identity_sha256"] is None


def test_published_sources_aggregate_across_actions_without_duplicate_counts():
    events = [
        ("graph_selected_sources_v2", {"chunk_ids": (101, 103)}),
        ("graph_selected_sources_v2", {"chunk_ids": (103, 104)}),
    ]
    result = summarize(events, [{"chunk_id": 103, "text": "private"}], [])
    assert result["graph_source_provenance"]["selected_count"] == 3
    assert result["graph_source_provenance"]["packet_count"] == 1


def test_invalid_published_identity_cannot_create_positive_evidence():
    for values in ((True,), (0,), ("103",), (103, 103), tuple(range(1, 47))):
        result = summarize(
            [("graph_selected_sources_v2", {"chunk_ids": values})],
            [{"chunk_id": 103, "text": "private"}],
            [],
        )
        assert result["graph_source_provenance"]["status"] == "invalid"
        assert result["packet"][0]["graph_selected_source"] is None


def test_emitter_is_lazy_without_observer_and_does_not_change_application_flow():
    replay_observation.graph_selected_sources_v2(object())
    events = []
    with observe(lambda name, data: events.append((name, data))) as state:
        replay_observation.graph_selected_sources_v2((101, 103))
    assert not state.failed.is_set()
    assert events[0][0] == "graph_selected_sources_v2"
    assert events[0][1]["chunk_ids"] == (101, 103)
    assert set(events[0][1]) == {"at", "chunk_ids"}


def test_emitter_rejects_invalid_ids_and_contains_observer_failures():
    events = []
    with observe(lambda name, data: events.append((name, data))) as state:
        replay_observation.graph_selected_sources_v2((True,))
    assert state.failed.is_set() and not events

    def broken(*args):
        raise RuntimeError("private")

    with observe(broken) as state:
        replay_observation.graph_selected_sources_v2((103,))
    assert state.failed.is_set()
