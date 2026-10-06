"""Timings accept only fixed labels and bounded aggregate values."""

import pytest
import structlog.testing

from apps.knowledge_graph.retrieval.stage_diagnostics import graph_stage


@pytest.mark.parametrize("branch", ("direct", "extended", "shared"))
@pytest.mark.parametrize(
    "stage",
    (
        "ontology",
        "extraction",
        "entity_resolution",
        "topology",
        "materialization",
    ),
)
@pytest.mark.parametrize("end, expected", ((-1.0, 0), (0.012, 12), (90.0, 5000)))
def test_fixed_stage_timings_are_bounded(branch, stage, end, expected):
    times = iter((0.0, end))
    with structlog.testing.capture_logs() as events:
        with graph_stage(branch=branch, stage=stage, clock=lambda: next(times)):
            pass
    assert events == [
        {
            "event": "obs.rag.graph_stage",
            "log_level": "info",
            "branch": branch,
            "stage": stage,
            "elapsed_ms": expected,
        }
    ]


@pytest.mark.parametrize(
    "branch, stage",
    (
        ("private identifier", "ontology"),
        ("direct", "private source passage"),
        (None, "ontology"),
        ("direct", 1),
    ),
)
def test_unknown_timing_labels_fail_without_logging(branch, stage):
    with structlog.testing.capture_logs() as events:
        with pytest.raises(ValueError):
            with graph_stage(branch=branch, stage=stage, clock=lambda: 0.0):
                pytest.fail("invalid stage was admitted")
    assert events == []
