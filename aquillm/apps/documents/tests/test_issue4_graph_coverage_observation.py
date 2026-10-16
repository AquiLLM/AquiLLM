"""Private graph coverage survives observed and compact replay boundaries."""

import json
from itertools import product
from types import SimpleNamespace

import pytest

from apps.chat.evals.replay_observations import summarize
from apps.documents.tests.test_chunk_search_graph_overlay import _DOC_A, _snapshot
from apps.knowledge_graph.retrieval.graph_handoff import GraphCoverageV2
from lib.evidence_observation import observe
from lib.replay_coverage import coverage_reasons
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


class _CoverageTuple(tuple):
    pass


class _CoverageString(str):
    pass


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
        _unavailable_coverage(("not_ready", "not_ready")),
        _unavailable_coverage(("not_ready", "frontier_budget")),
        _unavailable_coverage(_CoverageTuple(("not_ready",))),
        _unavailable_coverage((_CoverageString("not_ready"),)),
        {**_unavailable_coverage(("not_ready",)), "selected_documents": True},
        {**_unavailable_coverage(("not_ready",)), "selected_documents": -1},
        {**_unavailable_coverage(("not_ready",)), "selected_documents": 10001},
        {**_unavailable_coverage(("not_ready",)), "selected_documents": "530"},
        {**_unavailable_coverage(("not_ready",)), "ready_documents": 531},
        {**_unavailable_coverage(("not_ready",)), "ready_documents": True},
        {**_unavailable_coverage(("not_ready",)), "ready_documents": -1},
        {**_unavailable_coverage(("not_ready",)), "ready_documents": 1},
        {**_unavailable_coverage(("not_ready",)), "status": _UntrustedCoverage()},
        {**_unavailable_coverage(("not_ready",)), "status": "unknown"},
        {
            **_unavailable_coverage(("not_ready",)),
            "status": _CoverageString("unavailable"),
        },
        {**_unavailable_coverage(("not_ready",)), "status": "complete"},
        {**_unavailable_coverage(("not_ready",)), "status": "partial"},
        {**_unavailable_coverage(()), "ready_documents": 1, "status": "partial"},
        {**_unavailable_coverage(("not_ready",)), "exhaustive": True},
        {**_unavailable_coverage(("not_ready",)), "exhaustive": 0},
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


def test_valid_partial_coverage_retains_its_reason_at_document_count_limit():
    events, state = _native_graph_observation(
        {
            "graph_coverage_v2": {
                **_unavailable_coverage(("not_ready",)),
                "selected_documents": 10000,
                "ready_documents": 1,
                "status": "partial",
            }
        }
    )
    assert not state.failed.is_set()
    assert events[0][1]["graph"]["reasons"] == ["not_ready"]
    assert summarize(events, [], [])["stages"][0]["graph"]["reasons"] == ["not_ready"]


def test_primitive_coverage_validation_matches_unchanged_application_contract():
    for selected, ready, status, reasons, exhaustive in product(
        (0, 1, 10000, 10001, True),
        (0, 1, 10001, True),
        ("complete", "partial", "unavailable", "unknown"),
        (
            (),
            ("not_ready",),
            ("frontier_budget", "not_ready"),
            ("not_ready", "frontier_budget"),
            ("not_ready", "not_ready"),
        ),
        (False, True, 0),
    ):
        coverage = dict(
            selected_documents=selected,
            ready_documents=ready,
            status=status,
            reasons=reasons,
            exhaustive=exhaustive,
        )
        try:
            expected = GraphCoverageV2(**coverage).reasons
        except (TypeError, ValueError):
            expected = ()
        assert coverage_reasons(coverage) == expected, coverage


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
