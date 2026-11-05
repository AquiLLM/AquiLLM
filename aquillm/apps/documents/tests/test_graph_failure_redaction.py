"""Native graph failure labels survive normal replay without arbitrary payloads."""

import json

import pytest

from apps.chat.evals.replay_observations import summarize
from apps.documents.tests.test_issue4_graph_coverage_observation import _native_graph_observation


SAFE = {"version": "native-graph-failure-v1", "type": "database_error",
        "reason": "statement_cancelled", "phase": "pool_ordering", "owner_valid": True}


def test_native_failure_survives_observed_and_compact_replay():
    events, state = _native_graph_observation({"graph_status": "error", "graph_failure": SAFE})
    assert not state.failed.is_set()
    assert events[0][1]["graph"].get("failure") == SAFE
    assert summarize(events, [], [])["stages"][0]["graph"].get("failure") == SAFE


class Untrusted(dict):
    def get(self, *args):
        raise AssertionError("untrusted diagnostic accessed")
    def __str__(self):
        raise AssertionError("untrusted diagnostic stringified")


@pytest.mark.parametrize("payload", [
    None, Untrusted(SAFE), {**SAFE, "message": "private source text"},
    {**SAFE, "type": "private exception type"}, {**SAFE, "reason": "private SQL"},
    {**SAFE, "phase": "private path"}, {**SAFE, "owner_valid": 1},
    {**SAFE, "version": "unknown"}, {**SAFE, "reason": Untrusted()},
])
def test_untrusted_failure_is_dropped_at_both_replay_boundaries(payload):
    events, state = _native_graph_observation({"graph_status": "error", "graph_failure": payload})
    assert not state.failed.is_set()
    assert events[0][1]["graph"].get("failure") is None
    compact = summarize([("retrieval_stages", {"graph": {"failure": payload}})], [], [])
    assert compact["stages"][0]["graph"].get("failure") is None
    assert "private" not in json.dumps(compact)


def test_failure_logger_cannot_replace_the_original_outcome_or_stringify_payload(monkeypatch):
    from apps.knowledge_graph.retrieval import graph_failure_observations_v2 as diagnostic

    class Logger:
        def warning(self, event, **fields):
            assert event == "obs.rag.graph_failure"
            assert fields["reason"] == "unclassified"
            raise KeyboardInterrupt("broken logging sink")

    monkeypatch.setattr(diagnostic, "logger", Logger())
    observed = diagnostic.failure_diagnostics(ValueError(Untrusted()), "materialization", owner_valid=True)
    assert observed == {"graph_failure": {
        "version": "native-graph-failure-v1", "type": "value_error",
        "reason": "unclassified", "phase": "materialization", "owner_valid": True,
    }}


def test_malformed_database_exception_metadata_remains_passive():
    from django.db import OperationalError
    from apps.knowledge_graph.retrieval.graph_failure_observations_v2 import failure_diagnostics

    class Cause(Exception):
        @property
        def sqlstate(self):
            raise AssertionError("untrusted cause inspected")

    error = OperationalError("private SQL")
    error.__cause__ = Cause()
    assert failure_diagnostics(error, "pool_ordering", owner_valid=True) == {}
