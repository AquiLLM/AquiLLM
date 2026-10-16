"""Private graph coverage survives observed and compact replay boundaries."""

import json
from types import SimpleNamespace

import pytest

from apps.chat.evals.replay_observations import summarize
from apps.documents.tests.test_chunk_search_graph_overlay import _DOC_A, _snapshot
from lib.evidence_observation import observe
from lib.replay_observation import retrieval_stages


def _native_graph_observation(diagnostics, *, enabled=True, sink=None):
    row = SimpleNamespace(pk=18, doc_id=_DOC_A, content="baseline passage")
    events = []
    with observe(sink or (lambda name, data: events.append((name, data)))) as state:
        retrieval_stages(
            "test query",
            1,
            _snapshot(baseline=(row,)),
            SimpleNamespace(combined_candidates=None, graph_candidates=()),
            (row,),
            diagnostics,
            overlay_enabled=enabled,
            hybrid_pool=(),
            graph_seed_attempted=enabled,
            acquisition_schedule_observed="native_closure",
        )
    return events, state


def _unavailable_coverage(reasons):
    return {
        "selected_documents": 530,
        "ready_documents": 0,
        "status": "unavailable",
        "reasons": reasons,
        "exhaustive": False,
    }


def test_empty_native_handoff_preserves_mixed_coverage_in_compact_json():
    events, state = _native_graph_observation(
        {
            "graph_status": "error",
            "graph_failure_reason": "frontier_budget",
            "graph_coverage_v2": _unavailable_coverage(
                ("frontier_budget", "not_ready")
            ),
        }
    )
    assert not state.failed.is_set()
    graph = events[0][1]["graph"]
    assert graph["finalized_candidates"] == []
    assert graph["reasons"] == ["frontier_budget", "not_ready"]
    assert graph["failure_reason"] == "frontier_budget"
    persisted = json.loads(json.dumps(summarize(events, [], [])))
    compact = persisted["stages"][0]["graph"]
    assert compact["finalized_count"] == 0
    assert compact["reasons"] == ["frontier_budget", "not_ready"]
    assert compact["failure_reason"] == "frontier_budget"


@pytest.mark.parametrize(
    "diagnostics,enabled,reasons,failure_reason",
    [
        (
            {"graph_coverage_v2": _unavailable_coverage(("not_ready",))},
            True,
            ["not_ready"],
            None,
        ),
        ({}, True, None, None),
        ({}, False, None, None),
        (
            {"graph_direct_reason": "readiness_mismatch"},
            True,
            ["readiness_mismatch"],
            None,
        ),
        (
            {
                "graph_direct_reason": "readiness_mismatch",
                "graph_extended_reason": "readiness_mismatch",
                "graph_failure_reason": "frontier_budget",
                "graph_coverage_v2": _unavailable_coverage(
                    ("frontier_budget", "not_ready")
                ),
            },
            True,
            ["readiness_mismatch", "frontier_budget", "not_ready"],
            "frontier_budget",
        ),
        (
            {
                "graph_failure_reason": "deadline",
                "graph_coverage_v2": _unavailable_coverage(
                    (
                        "deadline",
                        "evidence_budget",
                        "frontier_budget",
                        "incompatible_generation",
                        "not_ready",
                    )
                ),
            },
            True,
            [
                "deadline",
                "evidence_budget",
                "frontier_budget",
                "incompatible_generation",
                "not_ready",
            ],
            "deadline",
        ),
        ({"graph_failure_reason": "not_ready"}, True, None, None),
    ],
)
def test_graph_reason_order_and_legacy_disabled_observations(
    diagnostics, enabled, reasons, failure_reason
):
    events, state = _native_graph_observation(diagnostics, enabled=enabled)
    assert not state.failed.is_set()
    graph = events[0][1]["graph"]
    assert graph["reasons"] == reasons
    assert graph["failure_reason"] == failure_reason
    assert graph["finalized_candidates"] == []
    compact = json.loads(json.dumps(summarize(events, [], [])))["stages"][0]["graph"]
    assert compact["reasons"] == reasons
    assert compact["failure_reason"] == failure_reason


class _UntrustedCoverage:
    def __str__(self):
        raise AssertionError("untrusted coverage stringified")

    def __deepcopy__(self, _memo):
        raise AssertionError("untrusted coverage copied")


class _CoverageDict(dict):
    def get(self, *_args):
        raise AssertionError("non-exact coverage inspected")


@pytest.mark.parametrize(
    "coverage",
    [
        None,
        _UntrustedCoverage(),
        _CoverageDict(_unavailable_coverage(("not_ready",))),
        {"reasons": ("not_ready",)},
        _unavailable_coverage(["not_ready"]),
        _unavailable_coverage("not_ready"),
        _unavailable_coverage(("unknown private reason",)),
        _unavailable_coverage((_UntrustedCoverage(),)),
        _unavailable_coverage(("not_ready",) * 6),
        {**_unavailable_coverage(("not_ready",)), "selected_documents": True},
        {**_unavailable_coverage(("not_ready",)), "ready_documents": 531},
        {**_unavailable_coverage(("not_ready",)), "status": _UntrustedCoverage()},
        {**_unavailable_coverage(("not_ready",)), "exhaustive": True},
        {**_unavailable_coverage(("not_ready",)), "extra": _UntrustedCoverage()},
    ],
)
def test_malformed_coverage_is_ignored_without_copying_or_stringifying(coverage):
    events, state = _native_graph_observation(
        {"graph_failure_reason": "deadline", "graph_coverage_v2": coverage}
    )
    assert not state.failed.is_set()
    graph = events[0][1]["graph"]
    assert graph["reasons"] == ["deadline"]
    assert graph["failure_reason"] == "deadline"
    compact = json.loads(json.dumps(summarize(events, [], [])))["stages"][0]["graph"]
    assert compact["reasons"] == ["deadline"]
    assert compact["failure_reason"] == "deadline"


@pytest.mark.parametrize("failure_reason", [None, "not_ready", 1, _UntrustedCoverage()])
def test_compact_failure_reason_rejects_unsupported_or_nonexact_values(failure_reason):
    compact = summarize(
        [("retrieval_stages", {"graph": {"failure_reason": failure_reason}})], [], []
    )["stages"][0]["graph"]
    assert compact["failure_reason"] is None
    json.dumps(compact)


def test_coverage_observation_sink_failure_remains_passive():
    def fail(_name, _data):
        raise RuntimeError("sink failed")

    events, state = _native_graph_observation(
        {"graph_coverage_v2": _unavailable_coverage(("not_ready",))}, sink=fail
    )
    assert events == []
    assert state.failed.is_set()
