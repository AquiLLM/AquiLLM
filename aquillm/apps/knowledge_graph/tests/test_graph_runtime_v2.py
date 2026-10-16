"""The V2 runtime consumes the actual G4 loader, context, and one global budget."""

from dataclasses import fields, replace
from importlib import import_module, util
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.projection.frontier_adapter_v2 import (
    Neo4jCanonicalFrontierAdapterV2,
)
from apps.knowledge_graph.retrieval.topology.frontier_loader_v2 import (
    canonical_frontier_loader_v2,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
    FrontierIndexFixtureDriver,
)
from lib.knowledge_graph.retrieval_config import load_hybrid_retrieval_settings
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def module():
    name = "apps.knowledge_graph.retrieval.graph_runtime_v2"
    assert util.find_spec(name), "whole V2 runtime is missing"
    return import_module(name)


def setup_runtime(monkeypatch, *, query_cap=64):
    from contextlib import nullcontext

    monkeypatch.setattr(
        "apps.collections.services.bounded_retrieval_authorization.bounded_reauthorization_reads",
        lambda *args: nullcontext(),
    )
    m = module()
    execution = import_module(
        "apps.knowledge_graph.retrieval.graph_runtime_execution_v2"
    )
    f = FrontierFixture()
    settings = load_hybrid_retrieval_settings({})
    settings = SimpleNamespace(
        **{f.name: getattr(settings, f.name) for f in fields(settings)},
        graph_frontier_max_queries=query_cap,
    )
    settings.graph_remediation_mode = "v2"
    settings.graph_direct_enabled = True
    runtime = SimpleNamespace(
        settings=settings, codec=f.codec, clock=__import__("time").monotonic
    )
    runtime.authorization = SimpleNamespace(
        authorization_context_signature=f.selection.authorization_context_signature,
        selected_document_ids=frozenset(
            s.document.id for s in f.selection.representation_snapshot.selections
        ),
    )
    runtime._exact_request = lambda a, s: None
    runtime._direct_seeds = lambda **kwargs: f.seeds
    driver = FrontierIndexFixtureDriver(f.bundles)
    captured = []

    def loader(**kwargs):
        assert kwargs["selection"] is f.selection
        return canonical_frontier_loader_v2(
            selection=f.selection,
            codec=f.codec,
            adapter=Neo4jCanonicalFrontierAdapterV2(driver),
        )

    monkeypatch.setattr(
        execution, "select_ready_scope_v2", lambda **kwargs: f.selection
    )
    monkeypatch.setattr(execution, "create_canonical_frontier_loader_v2", loader)
    monkeypatch.setattr(
        execution,
        "group_request",
        lambda **kwargs: replace(
            f.request, query_digest=__import__("hashlib").sha256(b"query").hexdigest()
        ),
    )

    def hydrate(**kwargs):
        captured.append(kwargs)
        return ()

    monkeypatch.setattr(execution, "load_selection_inputs_v2", hydrate)
    runtime.topology_loader = SimpleNamespace(driver=object())
    return m, f, runtime, settings, driver, captured


def test_real_frontier_and_context_reach_global_selector(monkeypatch):
    m, f, runtime, settings, driver, captured = setup_runtime(monkeypatch)
    budget = TurnBudget(TurnLimits())
    result = m.run_graph_v2(
        query="query",
        runtime=runtime,
        authorization=runtime.authorization,
        representation_snapshot=f.selection.representation_snapshot,
        budget=budget,
        settings=settings,
        deadline=f.deadline,
    )
    assert result.selection is f.selection
    assert captured[0]["selected"] is f.selection
    assert captured[0]["budget"] is budget
    assert captured[0]["references"]
    assert result.handoff.coverage.exhaustive is False
    assert 0 < len(driver.calls) <= 64


def test_exhausted_frontier_has_truthful_zero_candidate_coverage(monkeypatch):
    m, f, runtime, settings, driver, captured = setup_runtime(monkeypatch, query_cap=1)
    result = m.run_graph_v2(
        query="query",
        runtime=runtime,
        authorization=runtime.authorization,
        representation_snapshot=f.selection.representation_snapshot,
        budget=TurnBudget(TurnLimits()),
        settings=settings,
        deadline=f.deadline,
    )
    assert not result.handoff.candidates
    assert "frontier_budget" in result.handoff.coverage.reasons
    assert len(driver.calls) <= 1


def test_runtime_rejects_shadow_in_production(monkeypatch):
    m, f, runtime, settings, _, _ = setup_runtime(monkeypatch)
    settings.graph_remediation_mode = "shadow"
    with pytest.raises(ValueError):
        m.run_graph_v2(
            query="query",
            runtime=runtime,
            authorization=runtime.authorization,
            representation_snapshot=f.selection.representation_snapshot,
            budget=TurnBudget(TurnLimits()),
            settings=settings,
            deadline=f.deadline,
        )


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("physical", "frontier_budget"),
        ("uncertain", "frontier_budget"),
        ("completed_response", "frontier_budget"),
        ("completed_window", "frontier_budget"),
        ("deadline", "deadline"),
        ("timeout_subclass", "deadline"),
        ("unknown", None),
        ("cancelled", None),
    ],
)
def test_hydration_failure_keeps_budget_and_deadline_reasons_distinct(
    monkeypatch, failure, expected
):
    from asyncio import CancelledError

    from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
        FrontierCompletedBudgetFailureV2,
    )
    from apps.knowledge_graph.retrieval.topology.frontier_window_contracts_v2 import (
        CompletedWindowBudgetFailure,
    )

    class OtherTimeout(TimeoutError):
        pass

    m, f, runtime, settings, driver, _ = setup_runtime(monkeypatch)
    now, frontiers = [runtime.clock()], []
    runtime.clock = lambda: now[0]
    factory = m.FrontierBudgetV2

    def frontier(**kwargs):
        value = factory(**kwargs)
        frontiers.append(value)
        return value

    def hydrate(**kwargs):
        value = frontiers[0]
        assert value.source_rows > 0
        if failure == "physical":
            value.reserve_read(value.max_source_rows)
        if failure == "uncertain":
            value.stop_uncertain()
            value.check_active()
        if failure == "deadline":
            now[0] = value.deadline
            value.check_active()
        raise {
            "completed_response": FrontierCompletedBudgetFailureV2(1),
            "completed_window": CompletedWindowBudgetFailure("a" * 64, 1, 1),
            "timeout_subclass": OtherTimeout("other timeout"),
            "unknown": RuntimeError("unknown failure"),
            "cancelled": CancelledError(),
        }[failure]

    monkeypatch.setattr(m, "FrontierBudgetV2", frontier)
    monkeypatch.setattr(
        "apps.knowledge_graph.retrieval.graph_runtime_execution_v2.load_selection_inputs_v2",
        hydrate,
    )

    def execute():
        return m.run_graph_v2(
            query="query", runtime=runtime, authorization=runtime.authorization,
            representation_snapshot=f.selection.representation_snapshot,
            budget=TurnBudget(TurnLimits(), clock=runtime.clock), settings=settings,
            deadline=f.deadline,
        )

    if expected is None:
        with pytest.raises(CancelledError if failure == "cancelled" else RuntimeError):
            execute()
    else:
        result = execute()
        assert result.handoff.candidates == ()
        assert expected in result.handoff.coverage.reasons
        if expected == "frontier_budget":
            assert "deadline" not in result.handoff.coverage.reasons
    assert frontiers[0].queries == len(driver.calls) <= 64
