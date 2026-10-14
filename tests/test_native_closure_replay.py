"""Pure replay contracts: unavailable pools and proven novel final attribution."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.chat.evals.replay_controls import replay_runtime
from apps.chat.evals.replay_observations import summarize
from lib.evidence_observation import observe
from lib.replay_observation import retrieval_stages

TARGET = {"chunk_ids": [99], "source_sha256": "a" * 64}


def captured_stage(*, overlay=True, rows=(1, 26, 99), pool_rows=None, **kwargs):
    def row(pk):
        return SimpleNamespace(pk=pk, doc_id="private-doc", content="private-body")

    snapshot = SimpleNamespace(
        baseline_candidates=(row(1),),
        source_admission=SimpleNamespace(baseline_order=tuple(range(1, 31))),
        vector_results=(),
        trigram_results=(),
        exact_results=(),
        graph_seeds=(),
        graph_seed_error=False,
        effective_limits=None,
    )
    finalized = tuple(map(row, rows))
    pool = finalized if pool_rows is None else tuple(map(row, pool_rows))
    events = []
    with observe(lambda name, data: events.append((name, data))) as state:
        retrieval_stages(
            "private-query",
            15,
            snapshot,
            SimpleNamespace(combined_candidates=pool, graph_candidates=()),
            finalized,
            {},
            overlay_enabled=overlay,
            hybrid_pool=finalized,
            graph_seed_attempted=False,
            **kwargs,
        )
    assert not state.failed.is_set()
    return events[0]


@pytest.mark.parametrize(
    "stage",
    [
        {},
        {"materialized_union": None, "materialized_union_status": "unavailable"},
        {
            "materialized_union": [{"chunk_id": 99}],
            "materialized_union_status": "unavailable",
        },
    ],
)
def test_missing_or_unavailable_pool_is_not_observed_empty(stage):
    result = summarize([("retrieval_stages", stage)], [], [TARGET])
    assert result["stages"][0]["pool_count"] is None
    assert result["stages"][0]["materialized_union_status"] == "unavailable"
    assert result["targets"][0]["pool_ranks"] is None
    assert result["targets"][0]["pool_status"] == "unavailable"


def test_observed_empty_and_legacy_scored_pools_remain_observed():
    for rows, expected in (([], []), ([{"chunk_id": 99, "rank": 2}], [2])):
        result = summarize(
            [("retrieval_stages", {"materialized_union": rows})], [], [TARGET]
        )
        assert result["stages"][0]["pool_count"] == len(rows)
        assert result["targets"][0]["pool_ranks"] == expected
        assert result["targets"][0]["pool_status"] == "observed"


def test_no_stage_cannot_claim_execution_or_observed_empty_pool():
    result = summarize([], [], [TARGET])
    assert result["acquisition_schedule"]["status"] == "unavailable"
    assert result["targets"][0]["pool_ranks"] is None


def test_native_stage_omits_materialized_union_and_keeps_actual_schedule():
    event = captured_stage(
        acquisition_schedule_requested="native_closure",
        acquisition_schedule_observed="native_closure",
        acquisition_score_status="not_requested",
        materialized_union_status="unavailable",
        preclosure_pool_count=40,
    )
    assert event[1]["materialized_union"] is None
    assert event[1]["graph"]["candidates"] is None
    assert [row["chunk_id"] for row in event[1]["graph"]["finalized_candidates"]] == [
        99
    ]
    result = summarize([event], [], [TARGET])
    stage = result["stages"][0]
    assert stage["preclosure_pool_count"] == 40
    assert stage["pool_count"] is None
    assert stage["acquisition_schedule_observed"] == "native_closure"
    assert stage["acquisition_score_status"] == "not_requested"
    assert stage["graph"]["unique_count"] is None
    assert stage["graph"]["finalized_count"] == 1
    assert stage["graph"]["candidate_provenance"] == "finalized_novel"
    assert result["acquisition_schedule"]["observed"] == ["native_closure"]
    assert "private" not in json.dumps(result)


def test_fallback_schedule_is_distinct_from_requested_intent():
    event = captured_stage(
        acquisition_schedule_requested="native_closure",
        acquisition_schedule_observed="scored",
        acquisition_initial_ineligibility_reason="native_ineligible",
        acquisition_score_status="complete",
    )
    result = summarize([event], [], [])
    stage = result["stages"][0]
    assert stage["acquisition_schedule_requested"] == "native_closure"
    assert stage["acquisition_schedule_observed"] == "scored"
    assert stage["acquisition_initial_ineligibility_reason"] == "native_ineligible"
    assert result["acquisition_schedule"]["matches_requested"] is False


def test_novel_final_excludes_full_captured_baseline_and_requires_finalization():
    events = [
        ("graph_selected_sources_v2", {"chunk_ids": (26, 99, 100)}),
        captured_stage(),
    ]
    packet = [{"chunk_id": pk, "text": "private-body"} for pk in (26, 99, 100)]
    result = summarize(events, packet, [])
    assert result["graph_source_provenance"]["packet_count"] == 3
    assert result["graph_novel_final_provenance"] == {
        "status": "captured",
        "baseline_metadata_status": "captured",
        "baseline_metadata_count": 30,
        "packet_count": 1,
    }
    assert "private" not in json.dumps(result)
    assert "chunk_id" not in json.dumps(result)


@pytest.mark.parametrize(
    "events",
    [
        [],
        [("graph_selected_sources_v2", {"chunk_ids": (99,)})],
        [("retrieval_stages", {"post_rerank": [{"chunk_id": 99}]})],
    ],
)
def test_missing_novelty_evidence_is_unavailable(events):
    result = summarize(events, [{"chunk_id": 99}], [])
    assert result["graph_novel_final_provenance"]["status"] == "unavailable"
    assert result["graph_novel_final_provenance"]["packet_count"] is None


def test_novel_graph_below_handoff_prefix_cannot_count_in_final_packet():
    event = captured_stage(rows=tuple(range(1, 16)), pool_rows=(*range(1, 16), 99))
    # Graph source 99 was selected but lies below the finalized tool top-15 prefix.
    result = summarize(
        [("graph_selected_sources_v2", {"chunk_ids": (99,)}), event],
        [{"chunk_id": pk} for pk in range(1, 13)],
        [TARGET],
    )
    assert result["graph_novel_final_provenance"]["packet_count"] == 0
    assert result["targets"][0]["pool_ranks"] == [16]
    assert result["targets"][0]["rerank_ranks"] == []
    assert result["targets"][0]["packet_ranks"] == []


def test_graph_off_has_observed_zero_with_staged_evidence():
    result = summarize([captured_stage(overlay=False)], [{"chunk_id": 1}], [])
    assert result["graph_novel_final_provenance"]["packet_count"] == 0


def test_schedule_control_overrides_ambient_mode_and_restores_it(monkeypatch):
    monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", "native_closure")
    args = (
        SimpleNamespace(RAG_CACHE_ENABLED=True),
        SimpleNamespace(),
        SimpleNamespace(synthesize_from_evidence=None),
        None,
    )
    with replay_runtime("baseline", *args):
        assert os.environ["RAG_ACQUISITION_SCHEDULE"] == "scored"
    assert os.environ["RAG_ACQUISITION_SCHEDULE"] == "native_closure"
    with replay_runtime("current", *args, acquisition_schedule="native_closure"):
        assert os.environ["RAG_ACQUISITION_SCHEDULE"] == "native_closure"
    with pytest.raises(ValueError):
        with replay_runtime("current", *args, acquisition_schedule="invalid"):
            pass


def test_untrusted_new_fields_do_not_leak_to_summary():
    stage = {
        key: "private-body"
        for key in (
            "acquisition_schedule_requested",
            "acquisition_schedule_observed",
            "acquisition_initial_ineligibility_reason",
            "acquisition_score_status",
            "materialized_union_status",
            "preclosure_pool_count",
        )
    }
    assert "private" not in json.dumps(summarize([("retrieval_stages", stage)], [], []))


def test_legacy_stage_does_not_infer_executed_schedule_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", "native_closure")
    result = summarize([captured_stage()], [], [])
    assert result["acquisition_schedule"]["status"] == "unavailable"
    assert result["stages"][0]["acquisition_schedule_observed"] is None
    assert result["stages"][0]["acquisition_score_status"] is None


def test_one_unobserved_stage_makes_aggregate_pool_and_schedule_unavailable():
    events = [
        captured_stage(acquisition_schedule_observed="scored"),
        ("retrieval_stages", {}),
    ]
    result = summarize(events, [], [TARGET])
    assert result["targets"][0]["pool_ranks"] is None
    assert result["acquisition_schedule"]["observed"] is None


def test_novelty_excludes_baseline_seen_in_any_captured_stage():
    first = captured_stage()
    second = captured_stage()
    second[1]["baseline_metadata_ids"] = (99,)
    result = summarize(
        [("graph_selected_sources_v2", {"chunk_ids": (99,)}), first, second],
        [{"chunk_id": 99}],
        [],
    )
    assert result["graph_novel_final_provenance"]["baseline_metadata_count"] == 31
    assert result["graph_novel_final_provenance"]["packet_count"] == 0


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "native_replay_cli", Path(__file__).parents[1] / "scripts/replay_retrieval.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_cli_requires_explicit_current_treatment_for_native_schedule():
    cli = load_cli()
    base = [
        "--manifest",
        "questions.json",
        "--principal-id",
        "1",
        "--collection-ids",
        "1",
        "--output",
        "report.json",
    ]
    assert cli.parse_args(base).acquisition_schedule == "scored"
    args = cli.parse_args(
        base + ["--experiment", "current", "--acquisition-schedule", "native_closure"]
    )
    assert cli.selected_modes(args)["acquisition_schedule"] == "native_closure"
    for suffix in (
        ["--acquisition-schedule", "native_closure"],
        ["--experiment", "current", "--acquisition-schedule", "invalid"],
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(base + suffix)


def test_effective_config_reports_requested_schedule_version_and_reserve(monkeypatch):
    from django.core.exceptions import ImproperlyConfigured

    from apps.chat.evals import replay_controls

    for mode, version, reserve in (
        ("scored", None, None),
        ("native_closure", "native-closure-v1", 1250),
    ):
        monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", mode)
        report = replay_controls.effective_schedule_config()
        assert report["acquisition_schedule"] == mode
        assert report["acquisition_schedule_version"] == version
        assert report["native_closure_reserve_ms"] == reserve
    monkeypatch.setenv("RAG_ACQUISITION_SCHEDULE", "invalid")
    with pytest.raises(ImproperlyConfigured):
        replay_controls.effective_schedule_config()
