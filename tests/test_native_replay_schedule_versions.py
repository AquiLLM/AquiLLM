"""Replay preserves exact admitted schedule versions without accepting forged labels."""

import pytest

from apps.chat.evals.replay_observations import summarize


class _StringSubclass(str):
    pass


def summarized_stage(version):
    stage = {
        "acquisition_schedule_requested": "native_closure",
        "acquisition_schedule_observed": "native_closure",
        "acquisition_schedule_version": version,
        "acquisition_score_status": "not_requested",
        "materialized_union_status": "unavailable",
    }
    return summarize([("retrieval_stages", stage)], [], [])


@pytest.mark.parametrize("version", [
    "native-closure-v2", "native-closure-v3", "native-closure-v4", "native-closure-v5",
])
def test_replay_preserves_historical_and_current_native_schedule_versions(version):
    report = summarized_stage(version)
    stage = report["stages"][0]
    assert stage["acquisition_schedule_version"] == version
    assert report["acquisition_schedule"]["observed"] == ["native_closure"]
    assert stage["pool_count"] is None and stage["rerank_count"] is None


@pytest.mark.parametrize("version", [
    "", "native-closure-v6", "native-closure-v4-private",
    _StringSubclass("native-closure-v4"), {"label": "native-closure-v5"}, True,
])
def test_replay_rejects_unknown_or_forged_schedule_versions(version):
    stage = summarized_stage(version)["stages"][0]
    assert stage["acquisition_schedule_version"] is None
    assert stage["pool_count"] is None and stage["rerank_count"] is None
