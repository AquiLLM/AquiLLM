"""Shared deterministic fakes for hybrid graph integration tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from apps.collections.services.retrieval_authorization import (
    OpaquePrincipalReference,
    _make_test_reauthorization_capability,
    freeze_retrieval_authorization_context,
)
from apps.documents.services.chunk_search import (
    HybridGraphRetrievalDependencies,
    text_chunk_search,
)
from apps.documents.tests.test_chunk_search_graph_overlay import (
    _DOC_A,
    _DOC_B,
    _model,
    _snapshot,
)
from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchEnvelopeV1,
    BranchProvenanceV1,
    BranchSafeDiagnosticsV1,
    BranchStatusV1,
    DirectBranchResultV1,
    ExtendedBranchResultV1,
    branch_candidate_order_checksum,
)
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind

KEYS = tuple(character * 64 for character in "cdef12")


def chunk(identifier: int, document_id=_DOC_A):
    return SimpleNamespace(
        pk=identifier,
        doc_id=document_id,
        chunk_number=identifier - 1,
        content=f"chunk-{identifier}",
    )


class Policy:
    policy_version = "collection-view-v1"
    policy_checksum = KEYS[0]

    def __init__(self) -> None:
        self.rows = ((7, _DOC_A), (9, _DOC_B))

    def opaque_principal_reference(self, **_kwargs):
        return OpaquePrincipalReference(KEYS[1])

    def current_authorized_document_scope(self, **_kwargs):
        return self.rows


def authorization(
    policy: Policy,
    *,
    collection_ids=(7, 9),
    document_ids=(_DOC_A, _DOC_B),
):
    principal = object()
    return freeze_retrieval_authorization_context(
        principal=principal,
        database_alias="default",
        policy=policy,
        selected_collection_ids=collection_ids,
        selected_document_ids=document_ids,
        reauthorization_capability=_make_test_reauthorization_capability(
            principal=principal,
            policy=policy,
        ),
    )


def successful_branch(kind: HybridBranchKind, candidates):
    candidates = tuple(candidates)
    provenance = BranchProvenanceV1(
        kind,
        KEYS[2],
        KEYS[3],
        1,
        KEYS[4],
        KEYS[5],
        len(candidates),
        branch_candidate_order_checksum(candidates),
        125,
        1,
    )
    result_type = (
        DirectBranchResultV1
        if kind is HybridBranchKind.DIRECT
        else ExtendedBranchResultV1
    )
    return BranchEnvelopeV1(
        kind,
        BranchStatusV1.SUCCEEDED,
        result_type(candidates, provenance),
        None,
        BranchSafeDiagnosticsV1(1, 1, 0, len(candidates), 1),
    )


def hybrid_settings():
    return SimpleNamespace(
        memgraph_traversal_enabled=True,
        graph_direct_enabled=True,
        graph_extended_enabled=True,
        graph_overall_timeout_ms=300,
        graph_direct_timeout_ms=125,
        graph_extended_timeout_ms=125,
        graph_direct_max_candidates=20,
        graph_extended_max_candidates=20,
        graph_fusion_rrf_k=60,
    )


def selected_snapshot(*, baseline: tuple[object, ...]):
    return replace(
        _snapshot(baseline=baseline),
        documents=(
            SimpleNamespace(id=_DOC_A, collection_id=7),
            SimpleNamespace(id=_DOC_B, collection_id=9),
        ),
    )


def run_hybrid_test_search(
    monkeypatch,
    *,
    snapshot,
    authorization_context,
    runtime,
    materialize,
    top_k=1,
):
    from apps.documents.services import chunk_search

    rerank_inputs: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        chunk_search,
        "collect_hybrid_candidate_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr("aquillm.utils.get_embedding", lambda _query: (0.1, 0.2))
    monkeypatch.setattr(
        chunk_search,
        "rerank_chunks",
        lambda _model, _query, rows, _top_k: (
            rerank_inputs.append(tuple(row.pk for row in rows)) or tuple(rows)
        ),
    )
    monkeypatch.setattr(
        chunk_search,
        "_fallback_rerank",
        lambda _model, rows, _top_k: (
            rerank_inputs.append(tuple(row.pk for row in rows)) or tuple(rows)
        ),
    )
    model, _filters = _model([])
    returned = text_chunk_search(
        model,
        "query",
        top_k,
        list(snapshot.documents),
        authorization_context=authorization_context,
        hybrid_graph_dependencies=HybridGraphRetrievalDependencies(
            runtime=runtime,
            settings=hybrid_settings(),
            materialize=materialize,
        ),
    )
    return returned, rerank_inputs


__all__ = [
    "KEYS",
    "Policy",
    "authorization",
    "chunk",
    "hybrid_settings",
    "run_hybrid_test_search",
    "selected_snapshot",
    "successful_branch",
]
