"""Executed kernel policy is bound into request identity and actual observation."""

from dataclasses import replace
from importlib import import_module

import pytest

from apps.knowledge_graph.tests.test_graph_runtime_v2 import setup_runtime
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def invoke(m, f, runtime, settings):
    return m.run_graph_v2(
        query="query",
        runtime=runtime,
        authorization=runtime.authorization,
        representation_snapshot=f.selection.representation_snapshot,
        budget=TurnBudget(TurnLimits()),
        settings=settings,
        deadline=f.deadline,
    )


def test_same_config_is_ranked_and_bound_to_whole_request(monkeypatch):
    m, f, runtime, settings, _, _ = setup_runtime(monkeypatch)
    execution = import_module(
        "apps.knowledge_graph.retrieval.graph_runtime_execution_v2"
    )
    original = invoke(m, f, runtime, settings)
    factory = execution.frontier_config
    seen = []

    def changed(caps, resolver):
        config = replace(factory(caps, resolver), ppr_restart=0.25)
        seen.append(config)
        return config

    # Snapshot production must use the same changed effective policy.
    monkeypatch.setattr(execution, "frontier_config", changed)
    monkeypatch.setattr(
        "apps.knowledge_graph.projection.frontier_snapshot_v2.frontier_config", changed
    )
    second = invoke(m, f, runtime, settings)
    assert seen
    assert original.handoff.algorithm_signature != second.handoff.algorithm_signature
    assert (
        second.frontier_invocation.request_algorithm_signature
        == second.handoff.algorithm_signature
    )
    with pytest.raises(ValueError):
        replace(
            second,
            frontier_invocation=replace(
                second.frontier_invocation,
                request_algorithm_signature=original.handoff.algorithm_signature,
            ),
        )
    with pytest.raises(ValueError):
        replace(
            second,
            frontier_invocation=replace(
                second.frontier_invocation, request_algorithm_signature=None
            ),
        )


def test_failed_early_execution_emits_unknown_request_identity(monkeypatch):
    from lib.evidence_observation import observe

    m, f, runtime, settings, _, _ = setup_runtime(monkeypatch)

    def fail(**kwargs):
        raise ValueError("synthetic early capture failure")

    monkeypatch.setattr(
        "apps.knowledge_graph.retrieval.graph_runtime_execution_v2.select_ready_scope_v2",
        fail,
    )
    events = []
    with observe(lambda event, value: events.append((event, value))):
        with pytest.raises(ValueError):
            invoke(m, f, runtime, settings)
    assert events[-1][0] == "graph_frontier_budget"
    assert events[-1][1]["request_algorithm_signature"] is None
