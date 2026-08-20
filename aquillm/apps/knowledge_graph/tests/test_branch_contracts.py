from dataclasses import FrozenInstanceError, fields, replace

import pytest

from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchEnvelopeV1,
    BranchProvenanceV1,
    BranchSafeDiagnosticsV1,
    BranchStatusV1,
    DirectBranchFailureReason,
    DirectBranchRequestV1,
    DirectBranchResultV1,
    ExtendedBranchFailureReason,
    ExtendedBranchRequestV1,
    ExtendedBranchResultV1,
    GraphBranchCandidateV1,
    HybridBranchOutcomeV1,
    SharedBranchFailureReason,
    branch_candidate_order_checksum,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    AuthorizedProjectedDocumentV1,
    HybridBranchKind,
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    SelectedCollectionGenerationV1,
    TopologyCapsV1,
    TopologyDeadlineV1,
    projected_seed_checksum,
    ready_generation_bundle_checksum,
)

K = tuple(character * 64 for character in "123456789abcdef")


def _ready() -> ReadyGenerationBundleV1:
    generation = SelectedCollectionGenerationV1(
        K[0],
        K[1],
        K[2],
        K[3],
        K[14],
        "memgraph-schema-v1",
        "projection-v1",
        "key-v1",
        1,
        K[4],
        "resolver-v1",
        K[5],
        K[6],
        "embed-v1",
    )
    documents = (AuthorizedProjectedDocumentV1(K[7], K[0], K[1]),)
    checksum = ready_generation_bundle_checksum((generation,), documents, K[8])
    return ReadyGenerationBundleV1((generation,), documents, K[8], checksum)


def _request(kind: HybridBranchKind):
    seeds = (ProjectedSeedV1(K[9], 1.0),)
    values = (
        _ready(),
        seeds,
        projected_seed_checksum(seeds),
        TopologyCapsV1(kind, 32, 2, 200, 1000, 20),
        TopologyDeadlineV1(kind, 125.0, 100.0),
    )
    request_type = (
        DirectBranchRequestV1
        if kind is HybridBranchKind.DIRECT
        else ExtendedBranchRequestV1
    )
    return request_type(*values)


def _provenance(kind: HybridBranchKind) -> BranchProvenanceV1:
    candidates = (GraphBranchCandidateV1("a" * 64, 1, 0.5),)
    return BranchProvenanceV1(
        branch_kind=kind,
        ready_bundle_checksum=_ready().bundle_checksum,
        seed_checksum=projected_seed_checksum((ProjectedSeedV1(K[9], 1.0),)),
        seed_count=1,
        topology_snapshot_checksum=K[10],
        ppr_algorithm_signature=K[11],
        candidate_count=1,
        candidate_order_checksum=branch_candidate_order_checksum(candidates),
        deadline_ms=100,
        elapsed_ms=12,
    )


def _result(kind: HybridBranchKind):
    candidates = (GraphBranchCandidateV1("a" * 64, 1, 0.5),)
    result_type = (
        DirectBranchResultV1
        if kind is HybridBranchKind.DIRECT
        else ExtendedBranchResultV1
    )
    return result_type(candidates, _provenance(kind))


def _diagnostics(**changes: int) -> BranchSafeDiagnosticsV1:
    values = {
        "seed_count": 1,
        "node_count": 4,
        "edge_count": 3,
        "candidate_count": 1,
        "elapsed_ms": 12,
    }
    values.update(changes)
    return BranchSafeDiagnosticsV1(**values)


def _success(kind: HybridBranchKind) -> BranchEnvelopeV1:
    return BranchEnvelopeV1(
        branch_kind=kind,
        status=BranchStatusV1.SUCCEEDED,
        result=_result(kind),
        failure_reason=None,
        diagnostics=_diagnostics(),
    )


def test_closed_shared_direct_extended_failure_values_and_status() -> None:
    assert tuple(SharedBranchFailureReason) == (
        "readiness_mismatch",
        "authorization_context_invalid",
        "backend_authentication",
        "backend_unavailable",
        "backend_provenance_mismatch",
        "backend_schema_mismatch",
        "overall_deadline",
        "fusion_invalid",
    )
    assert tuple(DirectBranchFailureReason) == (
        "extractor_timeout",
        "extractor_auth",
        "extractor_provenance",
        "mixed_ontology",
        "direct_seed_invalid",
        "direct_no_seeds",
        "direct_embedding_unavailable",
        "direct_topology_timeout",
        "direct_topology_invalid",
        "direct_ppr_invalid",
    )
    assert tuple(ExtendedBranchFailureReason) == (
        "extended_seed_invalid",
        "extended_no_seeds",
        "extended_topology_timeout",
        "extended_topology_invalid",
        "extended_ppr_invalid",
    )
    assert tuple(BranchStatusV1) == ("succeeded", "failed")


def test_branch_requests_are_exact_kind_bound_and_text_free() -> None:
    direct = _request(HybridBranchKind.DIRECT)
    extended = _request(HybridBranchKind.EXTENDED)
    assert direct.caps.branch_kind is HybridBranchKind.DIRECT
    assert extended.caps.branch_kind is HybridBranchKind.EXTENDED
    assert not {
        "query",
        "text",
        "database_id",
        "document_id",
    } & {field.name for field in fields(DirectBranchRequestV1)}
    with pytest.raises(ValueError, match="direct"):
        replace(direct, caps=extended.caps, deadline=extended.deadline)
    with pytest.raises(TypeError):
        replace(direct, seeds=list(direct.seeds))
    with pytest.raises(TypeError):
        replace(direct, caps="not-caps")
    bad = (ProjectedSeedV1(K[9], 0.6), ProjectedSeedV1(K[10], 0.5))
    with pytest.raises(ValueError, match="mass"):
        replace(direct, seeds=bad, seed_checksum=projected_seed_checksum(bad))


def test_candidates_and_provenance_pin_rank_order_checksum_and_safe_aggregates() -> (
    None
):
    candidates = (GraphBranchCandidateV1("a" * 64, 1, 0.5),)
    assert branch_candidate_order_checksum(candidates) == (
        "b4f0587c92997bb652357b9886c293d1b6b0c8f30f0bd182609a91fdc3fb74fc"
    )
    direct = _result(HybridBranchKind.DIRECT)
    with pytest.raises(FrozenInstanceError):
        direct.candidates = ()  # type: ignore[misc]
    assert not hasattr(direct, "__dict__")
    with pytest.raises(ValueError, match="contiguous"):
        replace(direct, candidates=(replace(candidates[0], rank=2),))
    for rows in (
        (
            GraphBranchCandidateV1("a" * 64, 1, 0.4),
            GraphBranchCandidateV1("b" * 64, 2, 0.5),
        ),
        (
            GraphBranchCandidateV1("b" * 64, 1, 0.5),
            GraphBranchCandidateV1("a" * 64, 2, 0.5),
        ),
    ):
        with pytest.raises(ValueError, match="score|order"):
            replace(direct, candidates=rows)
    with pytest.raises(ValueError, match="checksum|count"):
        replace(direct, candidates=(replace(candidates[0], chunk_key="b" * 64),))
    with pytest.raises(TypeError):
        GraphBranchCandidateV1("a" * 64, 1, 1)
    with pytest.raises(ValueError):
        replace(_provenance(HybridBranchKind.DIRECT), elapsed_ms=101)
    assert not any(
        name.endswith("_id") or "text" in name
        for name in (field.name for field in fields(BranchSafeDiagnosticsV1))
    )


def test_envelope_distinguishes_local_failures_and_preserves_successful_sibling() -> (
    None
):
    direct_failure = BranchEnvelopeV1(
        HybridBranchKind.DIRECT,
        BranchStatusV1.FAILED,
        None,
        DirectBranchFailureReason.MIXED_ONTOLOGY,
        _diagnostics(candidate_count=0),
    )
    outcome = HybridBranchOutcomeV1(
        direct=direct_failure,
        extended=_success(HybridBranchKind.EXTENDED),
        shared_failure_reason=None,
    )
    assert outcome.extended.status is BranchStatusV1.SUCCEEDED
    with pytest.raises(TypeError):
        replace(
            direct_failure, failure_reason=ExtendedBranchFailureReason.EXTENDED_NO_SEEDS
        )
    with pytest.raises(ValueError, match="status"):
        replace(direct_failure, status=BranchStatusV1.SUCCEEDED)


def test_shared_failure_requires_both_branches_to_fail_with_same_reason() -> None:
    reason = SharedBranchFailureReason.BACKEND_UNAVAILABLE
    direct = BranchEnvelopeV1(
        HybridBranchKind.DIRECT,
        BranchStatusV1.FAILED,
        None,
        reason,
        _diagnostics(candidate_count=0),
    )
    extended = replace(direct, branch_kind=HybridBranchKind.EXTENDED)
    HybridBranchOutcomeV1(direct, extended, reason)
    with pytest.raises(ValueError, match="shared"):
        HybridBranchOutcomeV1(direct, _success(HybridBranchKind.EXTENDED), reason)
    with pytest.raises(TypeError):
        HybridBranchOutcomeV1(direct, extended, "backend_unavailable")
