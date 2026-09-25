from dataclasses import FrozenInstanceError, fields, replace
from math import fsum

import pytest

from apps.knowledge_graph.retrieval.topology.contracts import (
    AuthorizedProjectedDocumentV1,
    HybridBranchKind,
    ProjectedSeedV1,
    ProjectedTopologyLoader,
    ProjectedTopologyQueryDriver,
    ProjectedTopologyRequestV1,
    ReadyGenerationBundleV1,
    SelectedCollectionGenerationV1,
    TopologyCapsV1,
    TopologyDeadlineV1,
    TopologyFailureReason,
    TopologyLoadResultV1,
    projected_seed_checksum,
    ready_generation_bundle_checksum,
    validate_projected_seed_sequence,
)

K = tuple(character * 64 for character in "123456789abcdef")


def _generation() -> SelectedCollectionGenerationV1:
    return SelectedCollectionGenerationV1(
        collection_key=K[0],
        generation_key=K[1],
        active_artifact_key=K[2],
        projection_key=K[3],
        graph_checksum=K[14],
        schema_version="memgraph-schema-v1",
        projection_version="projection-v1",
        identifier_key_version="key-v1",
        membership_epoch=7,
        membership_checksum=K[4],
        resolver_version="resolver-v1",
        resolution_config_checksum=K[5],
        ontology_checksum=K[6],
        embedding_model_signature="embed-v1",
    )


def _ready() -> ReadyGenerationBundleV1:
    generations = (_generation(),)
    documents = (AuthorizedProjectedDocumentV1(K[7], K[0], K[1]),)
    return ReadyGenerationBundleV1(
        selected_generations=generations,
        authorized_documents=documents,
        authorization_context_signature=K[8],
        bundle_checksum=(
            "0a72bdf2b473ddc41f32d9e218972355ba0e5e9041caca2c8df359d4d55f6d58"
        ),
    )


def _caps(kind: HybridBranchKind = HybridBranchKind.DIRECT) -> TopologyCapsV1:
    return TopologyCapsV1(kind, 32, 2, 200, 1000, 20)


def _deadline(kind: HybridBranchKind = HybridBranchKind.DIRECT) -> TopologyDeadlineV1:
    return TopologyDeadlineV1(kind, 125.0, 100.0)


def test_ready_generation_checksum_vector_fields_and_scope_closure() -> None:
    ready = _ready()
    assert (
        ready_generation_bundle_checksum(
            ready.selected_generations,
            ready.authorized_documents,
            ready.authorization_context_signature,
        )
        == ready.bundle_checksum
    )
    assert ReadyGenerationBundleV1.__module__.endswith("topology.contracts")
    assert tuple(field.name for field in fields(SelectedCollectionGenerationV1))[
        :5
    ] == (
        "collection_key",
        "generation_key",
        "active_artifact_key",
        "projection_key",
        "graph_checksum",
    )
    assert tuple(field.name for field in fields(ReadyGenerationBundleV1)) == (
        "selected_generations",
        "authorized_documents",
        "authorization_context_signature",
        "bundle_checksum",
    )
    with pytest.raises(FrozenInstanceError):
        ready.bundle_checksum = K[9]  # type: ignore[misc]
    assert not hasattr(ready, "__dict__")
    with pytest.raises(ValueError, match="checksum"):
        replace(ready, bundle_checksum=K[9])
    changed = (replace(ready.selected_generations[0], graph_checksum=K[13]),)
    assert (
        ready_generation_bundle_checksum(
            changed, ready.authorized_documents, ready.authorization_context_signature
        )
        != ready.bundle_checksum
    )
    with pytest.raises((TypeError, ValueError)):
        replace(ready.selected_generations[0], graph_checksum=K[10].upper())
    with pytest.raises(ValueError, match="closure"):
        replace(
            ready,
            authorized_documents=(
                replace(ready.authorized_documents[0], generation_key=K[9]),
            ),
        )


def test_generation_document_order_uniqueness_and_exact_builtin_types() -> None:
    ready = _ready()
    second = replace(
        _generation(),
        collection_key=K[9],
        generation_key=K[10],
        projection_key=K[11],
        active_artifact_key=K[13],
    )
    document = AuthorizedProjectedDocumentV1(K[12], K[9], K[10])
    checksum = ready_generation_bundle_checksum(
        (ready.selected_generations[0], second),
        (ready.authorized_documents[0], document),
        ready.authorization_context_signature,
    )
    ReadyGenerationBundleV1(
        (ready.selected_generations[0], second),
        (ready.authorized_documents[0], document),
        ready.authorization_context_signature,
        checksum,
    )
    with pytest.raises(ValueError, match="sorted"):
        ReadyGenerationBundleV1(
            (second, ready.selected_generations[0]),
            (ready.authorized_documents[0], document),
            ready.authorization_context_signature,
            checksum,
        )
    bad_mass = (ProjectedSeedV1(K[9], 0.6), ProjectedSeedV1(K[10], 0.5))
    with pytest.raises(ValueError, match="mass"):
        validate_projected_seed_sequence(
            bad_mass,
            maximum=2,
            expected_checksum=projected_seed_checksum(bad_mass),
        )
    with pytest.raises(TypeError):
        replace(_generation(), membership_epoch=True)
    with pytest.raises((TypeError, ValueError)):
        replace(_generation(), membership_checksum=K[10].upper())
    with pytest.raises(ValueError):
        replace(_generation(), resolver_version=" resolver-v1")


def test_selected_generations_require_one_projection_and_resolver_signature() -> None:
    ready = _ready()
    second = replace(
        _generation(),
        collection_key=K[9],
        generation_key=K[10],
        projection_key=K[11],
        active_artifact_key=K[12],
        resolver_version="resolver-v2",
    )
    documents = (
        ready.authorized_documents[0],
        AuthorizedProjectedDocumentV1(K[13], K[9], K[10]),
    )
    checksum = ready_generation_bundle_checksum(
        (ready.selected_generations[0], second),
        documents,
        ready.authorization_context_signature,
    )
    with pytest.raises(ValueError, match="signature"):
        ReadyGenerationBundleV1(
            (ready.selected_generations[0], second),
            documents,
            ready.authorization_context_signature,
            checksum,
        )


def test_projected_seeds_normalize_and_request_binds_branch_caps_deadline() -> None:
    seeds = (ProjectedSeedV1(K[9], 1.0),)
    checksum = projected_seed_checksum(seeds)
    assert (
        checksum == "647e15256698dafbb6ee760b0cd64ec3bbc1338552ec2cc0ebf73545532718d7"
    )
    request = ProjectedTopologyRequestV1(
        ready=_ready(),
        seeds=seeds,
        seed_checksum=checksum,
        caps=_caps(),
        deadline=_deadline(),
    )
    assert fsum(seed.mass for seed in request.seeds) == 1.0
    with pytest.raises(ValueError, match="branch"):
        replace(request, deadline=_deadline(HybridBranchKind.EXTENDED))
    with pytest.raises(ValueError, match="checksum"):
        replace(request, seed_checksum=K[0])
    with pytest.raises(TypeError):
        ProjectedSeedV1(K[9], 1)
    with pytest.raises(ValueError, match="sorted"):
        replace(
            request,
            seeds=(ProjectedSeedV1(K[10], 0.5), ProjectedSeedV1(K[9], 0.5)),
            seed_checksum=projected_seed_checksum(
                (ProjectedSeedV1(K[10], 0.5), ProjectedSeedV1(K[9], 0.5))
            ),
        )


def test_branch_caps_deadlines_and_failure_values_are_closed() -> None:
    assert tuple(HybridBranchKind) == ("direct", "extended")
    assert tuple(TopologyFailureReason) == (
        "readiness_mismatch",
        "authorization_context_invalid",
        "backend_authentication",
        "backend_unavailable",
        "backend_provenance_mismatch",
        "backend_schema_mismatch",
        "overall_deadline",
        "direct_topology_timeout",
        "direct_topology_invalid",
        "extended_topology_timeout",
        "extended_topology_invalid",
    )
    for changes in (
        {"max_depth": 3},
        {"max_nodes": 201},
        {"max_edges": 1001},
        {"max_seeds": True},
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(_caps(), **changes)
    for values in ((float("nan"), 1.0), (1.0, 2.0), (1, 1.0)):
        with pytest.raises((TypeError, ValueError)):
            TopologyDeadlineV1(HybridBranchKind.DIRECT, *values)


def test_provider_neutral_protocols_and_safe_result_shape() -> None:
    class Driver:
        def execute_read(
            self, *, query, parameters, deadline, max_records
        ):  # pragma: no cover - structural check
            return ()

    class Loader:
        def load(
            self, *, ready, seeds, caps, deadline
        ):  # pragma: no cover - structural check
            raise NotImplementedError

    assert isinstance(Driver(), ProjectedTopologyQueryDriver)
    assert isinstance(Loader(), ProjectedTopologyLoader)
    result = TopologyLoadResultV1(
        branch_kind=HybridBranchKind.DIRECT,
        ready_bundle_checksum=_ready().bundle_checksum,
        seed_checksum=projected_seed_checksum((ProjectedSeedV1(K[9], 1.0),)),
        snapshot_checksum=None,
        node_count=0,
        edge_count=0,
        elapsed_ms=12,
        snapshot=None,
        failure_reason=TopologyFailureReason.BACKEND_UNAVAILABLE,
    )
    assert result.failure_reason is TopologyFailureReason.BACKEND_UNAVAILABLE
    replace(result, failure_reason=TopologyFailureReason.DIRECT_TOPOLOGY_TIMEOUT)
    replace(
        result,
        branch_kind=HybridBranchKind.EXTENDED,
        failure_reason=TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID,
    )
    assert not {
        "query",
        "text",
        "database_id",
        "document_id",
    } & {field.name for field in fields(TopologyLoadResultV1)}
    with pytest.raises(TypeError):
        replace(result, failure_reason="backend_unavailable")
    replace(result, failure_reason=TopologyFailureReason.OVERALL_DEADLINE)
    for kind, reason in (
        (HybridBranchKind.DIRECT, TopologyFailureReason.EXTENDED_TOPOLOGY_TIMEOUT),
        (HybridBranchKind.EXTENDED, TopologyFailureReason.DIRECT_TOPOLOGY_INVALID),
    ):
        with pytest.raises(ValueError, match="branch"):
            replace(result, branch_kind=kind, failure_reason=reason)
