"""The supported non-native V2 route honors the same finite graph allowance."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from apps.documents.services import hybrid_graph_v2 as graph_pool
from apps.knowledge_graph.retrieval import graph_candidate_operation_v2 as owner
from lib.retrieval import TurnBudget, TurnLimits


@pytest.mark.parametrize("timeout,elapsed,remaining", [
    (20000, 0, 20), (20000, 23, 7), (5000, 0, 5),
    (20001, 0, None), (True, 0, None), (0, 0, None),
])
def test_non_native_v2_deadline_accepts_configured_time_and_clamps_turn(
    monkeypatch, timeout, elapsed, remaining,
):
    now = [100.0]
    budget = TurnBudget(TurnLimits(retrieval_ms=30000), clock=lambda: now[0])
    now[0] += elapsed
    authorization = object()
    scope = SimpleNamespace(
        runtime=SimpleNamespace(budget=budget), authorization=authorization,
        representation=object(), requested_document_ids=(),
        database=nullcontext, validate=lambda: None,
    )
    settings = SimpleNamespace(
        graph_remediation_mode="v2", graph_overall_timeout_ms=timeout,
    )
    seen = []

    def run_v2(**kwargs):
        seen.append(kwargs)
        # This regression stops at the real runtime boundary; graph I/O is forbidden.
        raise RuntimeError("test runtime boundary reached")

    runtime = SimpleNamespace(run_v2=run_v2)
    dependencies = SimpleNamespace(settings=settings, runtime=runtime)
    snapshot = SimpleNamespace(
        graph_seeds=(), baseline_candidates=(), documents=(), source_admission=None,
    )
    monkeypatch.setattr(graph_pool, "perf_counter", lambda: now[0])
    monkeypatch.setattr(graph_pool, "current_scope", lambda: scope)
    monkeypatch.setattr(owner, "current_scope", lambda: scope)
    monkeypatch.setattr(graph_pool, "validate_requested_authorization", lambda *_: None)
    monkeypatch.setattr(graph_pool, "validate_rows", lambda rows, **_: rows)
    monkeypatch.setattr(graph_pool, "reauthorized_baseline", lambda *_: ((), True))
    # Exact native eligibility is actually evaluated, not replaced with a mock.
    assert owner._capture_closed_graph_attempt(
        snapshot, "q", authorization, dependencies, diagnostics=lambda *_: {},
    ) is None
    rows, _ = graph_pool.hybrid_graph_candidate_pool_v2(
        snapshot, "q", authorization, dependencies,
    )
    assert rows == ()
    if remaining is None:
        assert seen == []
    else:
        assert len(seen) == 1
        assert seen[0]["budget"] is budget
        assert seen[0]["authorization"] is authorization
        assert seen[0]["deadline"] == now[0] + remaining
