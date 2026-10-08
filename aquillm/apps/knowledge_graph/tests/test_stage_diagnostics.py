"""Timings accept only fixed labels and bounded aggregate values."""

import json
import os
import subprocess
import sys
from pathlib import Path

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


def test_stage_event_reaches_production_handlers_and_is_redacted():
    # A separate process preserves pytest's logging state and runs the real
    # filter_by_level/ProcessorFormatter pipeline instead of capture_logs.
    environment = dict(os.environ)
    environment.pop("LOKI_PUSH_URL", None)
    environment["DJANGO_DEBUG"] = "0"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import logging.config
import structlog
from aquillm.settings_logging import LOGGING
from apps.knowledge_graph.retrieval.stage_diagnostics import graph_stage
logging.config.dictConfig(LOGGING)
structlog.contextvars.bind_contextvars(query='PRIVATE_QUERY', prompt='PRIVATE_PROMPT')
times = iter((0.0, 90.0))
try:
    with graph_stage(
        branch='direct', stage='entity_resolution', clock=lambda: next(times)
    ):
        raise RuntimeError('PRIVATE_EXCEPTION')
except RuntimeError:
    pass
""",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    events = [json.loads(line) for line in result.stderr.splitlines() if line.strip()]
    assert len(events) == 1
    assert events[0]["event"] == "obs.rag.graph_stage"
    assert events[0]["branch"] == "direct"
    assert events[0]["stage"] == "entity_resolution"
    assert events[0]["elapsed_ms"] == 5000
    assert events[0]["level"] == "info"
    assert "PRIVATE" not in result.stdout + result.stderr
