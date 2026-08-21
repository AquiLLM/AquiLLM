"""Production hybrid runtime scoring and branch envelope contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apps.documents.tests.hybrid_graph_test_support import Policy, authorization
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1,
    DirectBranchFailureReason,
    ExtendedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.production_extended import prepare_extended_branch
from apps.knowledge_graph.retrieval.production_runtime import (
    ProductionHybridBranchRuntime,
    graph_candidates,
)
from apps.knowledge_graph.retrieval.production_runtime_support import (
    ppr_failure_envelope,
)
from apps.knowledge_graph.retrieval.ready_scope import assemble_selected_ready_scope
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind
from apps.knowledge_graph.tests.projected_ppr_fixtures import key, projected_snapshot
from apps.knowledge_graph.tests.test_ready_scope import (
    _DOC_A,
    _DOC_B,
    _authority,
)


def _ready_scope():
    return assemble_selected_ready_scope(
        authorization=authorization(Policy()),
        authorities=(_authority(7, _DOC_A, "1"), _authority(9, _DOC_B, "2")),
        codec=HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1"),
    )


def test_graph_candidates_overlay_identity_scores_on_mentions_and_relation_evidence():
    snapshot, _config = projected_snapshot(edges=(("a", "b"),))
    candidates = graph_candidates(
        snapshot=snapshot,
        identity_scores=((key("a"), 0.8), (key("b"), 0.2), (key("c"), 0.0)),
        maximum=20,
    )

    assert candidates
    assert tuple(row.rank for row in candidates) == tuple(range(1, len(candidates) + 1))
    assert candidates == tuple(
        sorted(candidates, key=lambda row: (-row.score, row.chunk_key))
    )
    assert all(0.0 <= row.score <= 1.0 for row in candidates)
    assert len({row.chunk_key for row in candidates}) == len(candidates)


def test_ppr_failure_envelope_retains_bounded_topology_diagnostics():
    snapshot, _config = projected_snapshot(edges=(("a", "b"),))

    envelope = ppr_failure_envelope(
        HybridBranchKind.DIRECT,
        DirectBranchFailureReason.DIRECT_PPR_INVALID,
        seed_count=2,
        snapshot=snapshot,
        elapsed_ms=7,
    )

    assert envelope.status is BranchStatusV1.FAILED
    assert envelope.failure_reason is DirectBranchFailureReason.DIRECT_PPR_INVALID
    assert envelope.diagnostics.seed_count == 2
    assert envelope.diagnostics.node_count == len(snapshot.identity_keys)
    assert envelope.diagnostics.edge_count == len(snapshot.relation_groups)
    assert envelope.diagnostics.elapsed_ms == 7


def test_shared_readiness_load_fails_if_deadline_expires_during_database_read():
    scope = _ready_scope()
    times = iter((0.0, 2.0))
    runtime = ProductionHybridBranchRuntime(
        authorization=authorization(Policy()),
        settings=object(),
        topology_loader=object(),
        codec=object(),
        scope_loader=lambda **_kwargs: scope,
        clock=lambda: next(times),
    )

    with pytest.raises(TimeoutError, match="deadline"):
        runtime.prepare_shared(
            authorization=runtime.authorization,
            settings=runtime.settings,
            deadline=1.0,
        )


def test_extended_seed_loading_fails_closed_before_database_io_after_deadline():
    scope = _ready_scope()
    runtime = SimpleNamespace(
        authorization=authorization(Policy()),
        clock=lambda: 2.0,
        _exact_request=lambda _authorization, _settings: None,
        _shared_scope=lambda _shared: scope,
    )
    settings = SimpleNamespace(
        graph_extended_enabled=True,
        graph_extended_max_seeds=3,
    )

    outcome = prepare_extended_branch(
        runtime,
        baseline=SimpleNamespace(
            graph_seeds=(object(),), baseline_candidates=()
        ),
        shared=object(),
        authorization=runtime.authorization,
        settings=settings,
        deadline=1.0,
    )

    assert outcome is ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT
