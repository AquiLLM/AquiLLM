"""Serving mode chooses one PageRank computation per branch."""

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

from apps.documents.tests.hybrid_graph_test_support import Policy, authorization
from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1,
)
from apps.knowledge_graph.retrieval.ppr import canonical_algorithm_json
from apps.knowledge_graph.retrieval.ppr_seed_support import (
    PreparedPPRSeedsV1,
    summarize_extended_support,
)
from apps.knowledge_graph.retrieval.production_ppr_policy import rank_projected_for_mode
from apps.knowledge_graph.retrieval.production_runtime import (
    ProductionHybridBranchRuntime,
)
from apps.knowledge_graph.retrieval.production_runtime_support import (
    ProductionSharedScopeV1,
    graph_candidates,
    success_envelope,
)
from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
from apps.knowledge_graph.retrieval.projected_types import projected_snapshot_checksum
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
)
from apps.knowledge_graph.retrieval.types import GraphExpansionSeed
from apps.knowledge_graph.tests.projected_ppr_fixtures import key, projected_snapshot
from apps.knowledge_graph.tests.test_production_hybrid_runtime import _ready_scope


def _inputs():
    snapshot, config = projected_snapshot(edges=(("a", "b"),))
    config = replace(config, ppr_iterations=8)
    signature = sha256(
        b"ppr_projected_v1\0" + canonical_algorithm_json(config)
    ).hexdigest()
    snapshot = replace(
        snapshot, algorithm=replace(snapshot.algorithm, algorithm_signature=signature)
    )
    seeds = (ProjectedSeedV1(key("a"), 1.0),)
    support = summarize_extended_support(
        ranked_seeds=(GraphExpansionSeed(10, 1, 1.0),),
        mapped_chunk_ids=frozenset((10,)),
        vector_chunk_ids=(10,),
        trigram_chunk_ids=(10,),
        exact_chunk_ids=(),
        retained_identity_count=1,
        max_seeds=64,
    )
    return snapshot, config, seeds, PreparedPPRSeedsV1(seeds, support, "focused")


def test_fixed_shadow_and_adaptive_dispatch_preserve_expected_rankings(monkeypatch):
    from apps.knowledge_graph.retrieval import production_ppr_policy as dispatch

    snapshot, config, seeds, prepared = _inputs()
    calls = []
    legacy = dispatch.ppr_projected_v1
    adaptive = dispatch.execute_adaptive_projected_ppr

    def legacy_once(**kwargs):
        calls.append("legacy")
        return legacy(**kwargs)

    def adaptive_once(**kwargs):
        calls.append("adaptive")
        return adaptive(**kwargs)

    monkeypatch.setattr(dispatch, "ppr_projected_v1", legacy_once)
    monkeypatch.setattr(dispatch, "execute_adaptive_projected_ppr", adaptive_once)
    outcomes = {}
    for mode in ("fixed", "shadow", "adaptive"):
        outcomes[mode] = rank_projected_for_mode(
            snapshot=snapshot,
            seeds=seeds,
            config=config,
            prepared=None if mode == "fixed" else prepared,
            mode=mode,
            branch=HybridBranchKind.EXTENDED,
            deadline_check=lambda: None,
        )
    assert calls == ["legacy", "legacy", "adaptive"]
    assert outcomes["fixed"][0] == outcomes["shadow"][0]
    assert outcomes["fixed"][1] is outcomes["shadow"][1] is None
    assert outcomes["adaptive"][0] != outcomes["fixed"][0]
    assert outcomes["adaptive"][1] != snapshot.algorithm.algorithm_signature
    assert outcomes["fixed"][0] == ppr_projected_v1(
        snapshot=snapshot, seeds=seeds, config=config
    )


def test_success_envelope_overrides_only_execution_algorithm_identity():
    snapshot, config, seeds, prepared = _inputs()
    ranking, signature = rank_projected_for_mode(
        snapshot=snapshot,
        seeds=seeds,
        config=config,
        prepared=prepared,
        mode="adaptive",
        branch=HybridBranchKind.DIRECT,
        deadline_check=lambda: None,
    )
    candidates = graph_candidates(
        snapshot=snapshot, identity_scores=ranking.scores, maximum=20
    )
    shared = dict(
        ready=_ready_scope().ready,
        seeds=seeds,
        snapshot=snapshot,
        candidates=candidates,
        settings=SimpleNamespace(
            graph_direct_timeout_ms=125, graph_extended_timeout_ms=125
        ),
        elapsed_ms=7,
    )
    original = success_envelope(HybridBranchKind.DIRECT, **shared)
    adaptive = success_envelope(
        HybridBranchKind.DIRECT,
        **shared,
        execution_algorithm_signature=signature,
    )
    assert (
        original.result.provenance.ppr_algorithm_signature
        == snapshot.algorithm.algorithm_signature
    )
    assert adaptive.result.provenance.ppr_algorithm_signature == signature
    assert (
        original.result.provenance.topology_snapshot_checksum
        == adaptive.result.provenance.topology_snapshot_checksum
        == projected_snapshot_checksum(snapshot)
    )
    assert (
        original.result.provenance.seed_checksum
        == adaptive.result.provenance.seed_checksum
    )
    assert (
        original.result.provenance.ready_bundle_checksum
        == adaptive.result.provenance.ready_bundle_checksum
    )


def _runtime(mode, snapshot, *, clock=lambda: 0.0):
    scope = _ready_scope()
    auth = authorization(Policy())
    settings = SimpleNamespace(
        ppr_restart_mode=mode,
        graph_direct_enabled=True,
        graph_extended_enabled=True,
        graph_direct_max_seeds=64,
        graph_direct_max_depth=2,
        graph_direct_max_nodes=200,
        graph_direct_max_edges=1000,
        graph_direct_max_candidates=20,
        graph_extended_max_seeds=64,
        graph_extended_max_depth=2,
        graph_extended_max_nodes=200,
        graph_extended_max_edges=1000,
        graph_extended_max_candidates=20,
        graph_direct_timeout_ms=125,
        graph_extended_timeout_ms=125,
    )
    runtime = ProductionHybridBranchRuntime(
        authorization=auth,
        settings=settings,
        topology_loader=SimpleNamespace(load=lambda **_kwargs: snapshot),
        codec=object(),
        clock=clock,
    )
    runtime._shared = ProductionSharedScopeV1(scope)
    return runtime, runtime._shared, auth, settings


def test_direct_and_extended_adaptive_keep_branch_local_original_checksums(monkeypatch):
    from apps.knowledge_graph.retrieval import production_runtime as direct_module

    snapshot, _config, seeds, prepared = _inputs()
    monkeypatch.setattr(
        direct_module, "prepare_direct_with_policy", lambda *_args, **_kwargs: prepared
    )
    runtime, shared, auth, settings = _runtime("adaptive", snapshot)
    direct = runtime.run_direct(
        query="who", shared=shared, authorization=auth, settings=settings, deadline=1.0
    )
    extended = runtime.run_extended(
        prepared=prepared,
        shared=shared,
        authorization=auth,
        settings=settings,
        deadline=1.0,
    )
    assert direct.status is extended.status is BranchStatusV1.SUCCEEDED
    for envelope in (direct, extended):
        provenance = envelope.result.provenance
        assert provenance.topology_snapshot_checksum == projected_snapshot_checksum(
            snapshot
        )
        assert (
            provenance.ppr_algorithm_signature != snapshot.algorithm.algorithm_signature
        )
    assert (
        direct.result.provenance.ppr_algorithm_signature
        == extended.result.provenance.ppr_algorithm_signature
    )
