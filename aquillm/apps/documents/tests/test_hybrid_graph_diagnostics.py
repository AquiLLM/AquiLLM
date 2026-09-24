"""Safe fixed branch observability, including duplicate-only success."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from apps.documents.services.hybrid_graph_authorization import (
    HybridGraphRetrievalDependencies,
)
from apps.documents.services.hybrid_graph_orchestration import (
    hybrid_graph_candidate_pool,
)
from apps.documents.tests.hybrid_graph_test_support import (
    KEYS,
    Policy,
    authorization,
    chunk,
    hybrid_settings,
    selected_snapshot,
    successful_branch,
)
from apps.documents.tests.test_chunk_search_graph_overlay import _DOC_A
from apps.knowledge_graph.retrieval.branch_contracts import (
    ExtendedBranchFailureReason,
    GraphBranchCandidateV1,
)
from apps.knowledge_graph.retrieval.materialization import MaterializedGraphChunkV1
from apps.knowledge_graph.retrieval.scheduler_support import failed_branch
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind
from apps.knowledge_graph.tests.test_retrieval_branch_scheduler import _Runtime


def test_duplicate_success_is_distinct_from_timeout_and_event_is_redacted(monkeypatch):
    from apps.documents.services import hybrid_graph_diagnostics

    events = []
    monkeypatch.setattr(
        hybrid_graph_diagnostics,
        "logger",
        SimpleNamespace(info=lambda event, **fields: events.append((event, fields))),
    )
    baseline = chunk(1)
    runtime = _Runtime()
    runtime.direct = successful_branch(
        HybridBranchKind.DIRECT, (GraphBranchCandidateV1(KEYS[0], 1, 0.9),)
    )
    runtime.extended = failed_branch(
        HybridBranchKind.EXTENDED,
        ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT,
        seed_count=1,
        elapsed_ms=125,
    )
    rows, diagnostics = hybrid_graph_candidate_pool(
        selected_snapshot(baseline=(baseline,)),
        "SECRET query",
        authorization(Policy()),
        HybridGraphRetrievalDependencies(
            runtime,
            hybrid_settings(),
            lambda **kw: (MaterializedGraphChunkV1(KEYS[0], 1, _DOC_A, 0, baseline),),
        ),
    )
    assert rows == (baseline,)
    assert diagnostics["graph_direct_status"] == "succeeded_duplicates"
    assert diagnostics["graph_direct_duplicate_count"] == 1
    assert diagnostics["graph_extended_status"] == "failed"
    assert diagnostics["graph_extended_reason"] == "extended_topology_timeout"
    assert len(events) == 1
    payload = json.dumps(events)
    assert (
        "SECRET" not in payload
        and KEYS[0] not in payload
        and str(_DOC_A) not in payload
    )


@pytest.mark.parametrize("failure", [False, True])
def test_empty_and_shared_failure_paths_emit_one_fixed_event(monkeypatch, failure):
    from apps.documents.services import hybrid_graph_diagnostics
    from apps.knowledge_graph.retrieval.branch_contracts import (
        SharedBranchFailureReason,
    )
    from apps.knowledge_graph.retrieval.scheduler_support import SharedSchedulerFailure

    events = []
    monkeypatch.setattr(
        hybrid_graph_diagnostics,
        "logger",
        SimpleNamespace(info=lambda event, **fields: events.append(fields)),
    )
    runtime = _Runtime()
    runtime.direct = successful_branch(HybridBranchKind.DIRECT, ())
    runtime.extended = successful_branch(HybridBranchKind.EXTENDED, ())
    if failure:

        def unavailable(**kwargs):
            raise SharedSchedulerFailure(SharedBranchFailureReason.BACKEND_UNAVAILABLE)

        runtime.prepare_shared = unavailable
    rows, diagnostics = hybrid_graph_candidate_pool(
        replace(selected_snapshot(baseline=()), baseline_candidates=()),
        "secret",
        authorization(Policy()),
        HybridGraphRetrievalDependencies(runtime, hybrid_settings(), lambda **kw: ()),
    )
    assert rows == ()
    assert len(events) == 1
    assert diagnostics["graph_raw_count"] == 0
    assert diagnostics["graph_direct_status"] == (
        "failed" if failure else "succeeded_empty"
    )
    assert diagnostics["graph_extended_status"] == diagnostics["graph_direct_status"]
