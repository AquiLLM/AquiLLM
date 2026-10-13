"""Whole opt-in graph runtime: one capture, frontier, hydration and selector."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING

from apps.collections.services.retrieval_authorization import (
    RetrievalAuthorizationContext,
)
from apps.documents.services.representation_contracts import RepresentationSnapshotV1
from apps.documents.services.source_deadline import source_preparation_scope
from apps.documents.services.source_loading import (
    bounded_source_database,
)
from lib.retrieval.turn_budget import TurnBudget

from .graph_runtime_contracts_v2 import (
    GraphRuntimeEnvelopeV2,
)
from .production_runtime_support import topology_caps
from .topology.contracts import HybridBranchKind
from .topology.frontier_contracts_v2 import FrontierBudgetV2

if TYPE_CHECKING:
    from .production_runtime import ProductionHybridBranchRuntime


def _caps(settings):
    enabled = [
        kind
        for kind, active in (
            (HybridBranchKind.DIRECT, settings.graph_direct_enabled),
            (HybridBranchKind.EXTENDED, settings.graph_extended_enabled),
        )
        if active
    ]
    if not enabled:
        raise ValueError("V2 graph has no enabled seed branch")
    values = [topology_caps(settings, kind) for kind in enabled]
    return replace(
        values[0],
        **{
            name: min(getattr(c, name) for c in values)
            for name in (
                "max_seeds",
                "max_depth",
                "max_nodes",
                "max_edges",
                "max_results",
            )
        },
    )


def run_graph_v2(
    *,
    query: str,
    runtime: ProductionHybridBranchRuntime,
    authorization: RetrievalAuthorizationContext,
    representation_snapshot: RepresentationSnapshotV1,
    budget: TurnBudget,
    settings,
    deadline: float,
    baseline=None,
    requested_document_ids=None,
) -> GraphRuntimeEnvelopeV2:
    from .graph_runtime_execution_v2 import execute_graph_v2

    return _run_graph_v2(
        query=query,
        runtime=runtime,
        authorization=authorization,
        representation_snapshot=representation_snapshot,
        budget=budget,
        settings=settings,
        deadline=deadline,
        baseline=baseline,
        requested_document_ids=requested_document_ids,
        execute=execute_graph_v2,
    )


def _run_graph_v2(
    *,
    query,
    runtime,
    authorization,
    representation_snapshot,
    budget,
    settings,
    deadline,
    baseline,
    requested_document_ids,
    execute,
):
    """Private runner shared with the eager native operation; no permit escapes."""
    if getattr(settings, "graph_remediation_mode", "legacy") != "v2":
        raise ValueError("whole V2 runtime requires explicit v2 mode")
    if (
        type(budget) is not TurnBudget
        or type(deadline) not in (float, int)
        or not isfinite(deadline)
    ):
        raise TypeError("original budget and finite deadline required")
    runtime._exact_request(authorization, settings)
    clock = runtime.clock
    branch_ms = [
        getattr(settings, f"graph_{kind}_timeout_ms")
        for kind in ("direct", "extended")
        if getattr(settings, f"graph_{kind}_enabled")
    ]
    deadline = min(
        deadline,
        clock() + budget.remaining_ms() / 1000,
        clock() + min(branch_ms) / 1000,
    )
    caps = _caps(settings)
    frontier = FrontierBudgetV2(
        turn_budget=budget,
        deadline=deadline,
        clock=clock,
        max_identities=caps.max_nodes,
        max_edges=caps.max_edges,
        max_source_rows=getattr(settings, "graph_frontier_max_source_rows", 5000),
        max_queries=getattr(settings, "graph_frontier_max_queries", 64),
        max_wire_bytes=getattr(settings, "graph_frontier_max_wire_bytes", 1048576),
    )
    from .graph_frontier_observation_v2 import (
        capture_frontier_budget_v2,
        publish_frontier_budget_v2,
    )

    handoff = None
    try:
        selected, handoff = execute(
            query=query,
            runtime=runtime,
            authorization=authorization,
            representation_snapshot=representation_snapshot,
            budget=budget,
            settings=settings,
            deadline=deadline,
            baseline=baseline,
            requested_document_ids=requested_document_ids,
            caps=caps,
            frontier=frontier,
        )
    finally:
        observation = capture_frontier_budget_v2(frontier, max_depth=caps.max_depth)
        invocation = publish_frontier_budget_v2(
            observation,
            query=query,
            request_algorithm_signature=(
                handoff.algorithm_signature if handoff else None
            ),
            requested_document_ids=(
                authorization.selected_document_ids
                if requested_document_ids is None
                else requested_document_ids
            ),
        )
    return GraphRuntimeEnvelopeV2(selected, handoff, invocation)


def run_bound_graph_v2(**kwargs):
    """All SQL phases and final selection share the original absolute deadline."""
    from apps.documents.services.graph_source_runtime import graph_source_runtime

    runtime = graph_source_runtime()
    if (
        runtime is None
        or runtime.budget is not kwargs["budget"]
        or runtime.authorization is not kwargs["authorization"]
    ):
        raise ValueError("V2 requires the original source runtime")
    settings, clock = kwargs["settings"], kwargs["runtime"].clock
    limits = [
        getattr(settings, f"graph_{kind}_timeout_ms")
        for kind in ("direct", "extended")
        if getattr(settings, f"graph_{kind}_enabled")
    ]
    kwargs["deadline"] = min(
        kwargs["deadline"],
        clock() + min(limits) / 1000,
        clock() + kwargs["budget"].remaining_ms() / 1000,
    )
    with source_preparation_scope(
        kwargs["deadline"], kwargs["runtime"].clock, kwargs["budget"]
    ):
        with bounded_source_database(runtime, kwargs["authorization"].database_alias):
            return run_graph_v2(**kwargs)
