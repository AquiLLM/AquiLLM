"""Failure isolation and authorization revalidation contracts."""

from __future__ import annotations

import pytest
from django.test import override_settings

from apps.documents.tests.hybrid_graph_test_support import (
    KEYS,
    Policy,
    authorization,
    chunk,
    run_hybrid_test_search,
    selected_snapshot,
    successful_branch,
)
from apps.documents.tests.test_chunk_search_graph_overlay import (
    _DOC_A,
    _DOC_B,
    _snapshot,
)
from apps.knowledge_graph.retrieval.branch_contracts import (
    ExtendedBranchFailureReason,
    GraphBranchCandidateV1,
    SharedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.materialization import MaterializedGraphChunkV1
from apps.knowledge_graph.retrieval.scheduler_support import (
    SharedSchedulerFailure,
    failed_branch,
)
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind


@pytest.mark.parametrize(
    "reason",
    (
        SharedBranchFailureReason.READINESS_MISMATCH,
        SharedBranchFailureReason.BACKEND_UNAVAILABLE,
        SharedBranchFailureReason.OVERALL_DEADLINE,
    ),
)
@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
def test_shared_failure_discards_both_graph_branches(monkeypatch, reason) -> None:
    baseline = (chunk(1), chunk(2))
    snapshot = selected_snapshot(baseline=baseline)
    calls: list[str] = []

    class Runtime:
        def prepare_shared(self, **_kwargs):
            calls.append("shared")
            raise SharedSchedulerFailure(reason)

        def run_direct(self, **_kwargs):
            raise AssertionError

        def prepare_extended(self, **_kwargs):
            raise AssertionError

        def run_extended(self, **_kwargs):
            raise AssertionError

    returned, rerank_inputs = run_hybrid_test_search(
        monkeypatch,
        snapshot=snapshot,
        authorization_context=authorization(Policy()),
        runtime=Runtime(),
        materialize=lambda **_kwargs: calls.append("materialize"),
    )

    assert calls == ["shared"]
    assert rerank_inputs == [(1, 2)]
    assert tuple(row.pk for row in returned[2]) == (1, 2)
    assert returned[3]["graph_status"] == "error"
    assert returned[3]["graph_candidate_count"] == 0


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
def test_branch_local_failure_preserves_completed_direct_sibling(monkeypatch) -> None:
    baseline = (chunk(1), chunk(2))
    graph = chunk(3)
    snapshot = selected_snapshot(baseline=baseline)
    materialized_keys: list[tuple[str, ...]] = []

    class Runtime:
        def prepare_shared(self, **_kwargs):
            return object()

        def run_direct(self, **_kwargs):
            return successful_branch(
                HybridBranchKind.DIRECT,
                (GraphBranchCandidateV1(KEYS[0], 1, 0.9),),
            )

        def prepare_extended(self, **_kwargs):
            return object()

        def run_extended(self, **_kwargs):
            return failed_branch(
                HybridBranchKind.EXTENDED,
                ExtendedBranchFailureReason.EXTENDED_NO_SEEDS,
            )

    def materialize(*, chunk_keys, **_kwargs):
        materialized_keys.append(tuple(key.value for key in chunk_keys))
        return (MaterializedGraphChunkV1(KEYS[0], 3, _DOC_A, 2, graph),)

    returned, rerank_inputs = run_hybrid_test_search(
        monkeypatch,
        snapshot=snapshot,
        authorization_context=authorization(Policy()),
        runtime=Runtime(),
        materialize=materialize,
    )

    assert materialized_keys == [(KEYS[0],)]
    assert rerank_inputs == [(1, 2, 3)]
    assert tuple(row.pk for row in returned[2]) == (1, 2, 3)
    assert returned[3]["graph_status"] == "hit"
    assert returned[3]["graph_candidate_count"] == 1


@pytest.mark.parametrize("revocation_window", ("before", "during_materialization"))
@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
def test_revocation_discards_graph_and_filters_baseline_in_order(
    monkeypatch,
    revocation_window,
) -> None:
    baseline = (chunk(1), chunk(2, _DOC_B))
    graph = chunk(3)
    snapshot = selected_snapshot(baseline=baseline)
    policy = Policy()
    auth = authorization(policy)
    materialize_calls: list[str] = []

    class Runtime:
        def prepare_shared(self, **_kwargs):
            return object()

        def run_direct(self, **_kwargs):
            return successful_branch(
                HybridBranchKind.DIRECT,
                (GraphBranchCandidateV1(KEYS[0], 1, 0.9),),
            )

        def prepare_extended(self, **_kwargs):
            return object()

        def run_extended(self, **_kwargs):
            return failed_branch(
                HybridBranchKind.EXTENDED,
                ExtendedBranchFailureReason.EXTENDED_NO_SEEDS,
            )

    def materialize(**_kwargs):
        materialize_calls.append("materialize")
        if revocation_window == "during_materialization":
            policy.rows = ((7, _DOC_A),)
        return (MaterializedGraphChunkV1(KEYS[0], 3, _DOC_A, 2, graph),)

    if revocation_window == "before":
        policy.rows = ((7, _DOC_A),)
    returned, _rerank_inputs = run_hybrid_test_search(
        monkeypatch,
        snapshot=snapshot,
        authorization_context=auth,
        runtime=Runtime(),
        materialize=materialize,
        top_k=3,
    )

    assert materialize_calls == (
        [] if revocation_window == "before" else ["materialize"]
    )
    assert tuple(row.pk for row in returned[2]) == (1,)
    assert returned[3]["graph_status"] == "error"
    assert returned[3]["graph_candidate_count"] == 0


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
def test_new_grants_are_ignored_without_discarding_selected_graph(monkeypatch) -> None:
    baseline = (chunk(1),)
    graph = chunk(3)
    snapshot = _snapshot(baseline=baseline)
    policy = Policy()
    policy.rows = ((7, _DOC_A),)
    auth = authorization(policy, collection_ids=(7,), document_ids=(_DOC_A,))

    class Runtime:
        def prepare_shared(self, **_kwargs):
            return object()

        def run_direct(self, **_kwargs):
            return successful_branch(
                HybridBranchKind.DIRECT,
                (GraphBranchCandidateV1(KEYS[0], 1, 0.9),),
            )

        def prepare_extended(self, **_kwargs):
            return object()

        def run_extended(self, **_kwargs):
            return failed_branch(
                HybridBranchKind.EXTENDED,
                ExtendedBranchFailureReason.EXTENDED_NO_SEEDS,
            )

    def materialize(**_kwargs):
        policy.rows = ((7, _DOC_A), (9, _DOC_B))
        return (MaterializedGraphChunkV1(KEYS[0], 3, _DOC_A, 2, graph),)

    returned, rerank_inputs = run_hybrid_test_search(
        monkeypatch,
        snapshot=snapshot,
        authorization_context=auth,
        runtime=Runtime(),
        materialize=materialize,
    )

    assert rerank_inputs == [(1, 3)]
    assert tuple(row.pk for row in returned[2]) == (1, 3)
    assert returned[3]["graph_status"] == "hit"


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
def test_reauthorization_error_fails_closed_after_materialization(monkeypatch) -> None:
    baseline = (chunk(1),)
    graph = chunk(3)
    snapshot = _snapshot(baseline=baseline)
    policy = Policy()
    policy.rows = ((7, _DOC_A),)
    auth = authorization(policy, collection_ids=(7,), document_ids=(_DOC_A,))

    class Runtime:
        def prepare_shared(self, **_kwargs):
            return object()

        def run_direct(self, **_kwargs):
            return successful_branch(
                HybridBranchKind.DIRECT,
                (GraphBranchCandidateV1(KEYS[0], 1, 0.9),),
            )

        def prepare_extended(self, **_kwargs):
            return object()

        def run_extended(self, **_kwargs):
            return failed_branch(
                HybridBranchKind.EXTENDED,
                ExtendedBranchFailureReason.EXTENDED_NO_SEEDS,
            )

    def materialize(**_kwargs):
        def fail_reauthorization(**_inner_kwargs):
            raise RuntimeError("authorization backend unavailable")

        policy.current_authorized_document_scope = fail_reauthorization
        return (MaterializedGraphChunkV1(KEYS[0], 3, _DOC_A, 2, graph),)

    returned, rerank_inputs = run_hybrid_test_search(
        monkeypatch,
        snapshot=snapshot,
        authorization_context=auth,
        runtime=Runtime(),
        materialize=materialize,
    )

    assert rerank_inputs == [()]
    assert not returned[2]
    assert returned[3]["graph_status"] == "error"
