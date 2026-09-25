"""Branch-local policy failures and authorized seed preparation."""

from types import SimpleNamespace

import pytest

from apps.documents.tests.hybrid_graph_test_support import Policy, authorization
from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1,
    DirectBranchFailureReason,
    ExtendedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.production_extended import (
    prepare_extended_with_policy,
)
from apps.knowledge_graph.retrieval.scheduler import HybridGraphBranchScheduler
from apps.knowledge_graph.retrieval.types import GraphExpansionSeed
from apps.knowledge_graph.tests.projected_ppr_fixtures import key
from apps.knowledge_graph.tests.test_production_ppr_dispatch import _inputs, _runtime


def test_invalid_direct_policy_is_local_and_cooperative_timeout_is_local(monkeypatch):
    from apps.knowledge_graph.retrieval import production_runtime as direct_module

    snapshot, _config, _seeds, prepared = _inputs()
    monkeypatch.setattr(
        direct_module,
        "prepare_direct_with_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad policy")),
    )
    runtime, shared, auth, settings = _runtime("adaptive", snapshot)
    invalid = runtime.run_direct(
        query="who", shared=shared, authorization=auth, settings=settings, deadline=1.0
    )
    assert invalid.failure_reason is DirectBranchFailureReason.DIRECT_SEED_INVALID
    monkeypatch.setattr(
        direct_module, "prepare_direct_with_policy", lambda *_args, **_kwargs: prepared
    )
    runtime.clock = lambda: 1.0
    direct_timeout = runtime.run_direct(
        query="who", shared=shared, authorization=auth, settings=settings, deadline=1.0
    )
    extended_timeout = runtime.run_extended(
        prepared=prepared,
        shared=shared,
        authorization=auth,
        settings=settings,
        deadline=1.0,
    )
    assert direct_timeout.failure_reason is DirectBranchFailureReason.EXTRACTOR_TIMEOUT
    assert (
        extended_timeout.failure_reason
        is ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT
    )


def test_invalid_prepared_policy_shape_fails_locally_before_topology(monkeypatch):
    from apps.knowledge_graph.retrieval import production_runtime as direct_module

    snapshot, _config, seeds, _prepared = _inputs()
    runtime, shared, auth, settings = _runtime("adaptive", snapshot)
    monkeypatch.setattr(
        direct_module, "prepare_direct_with_policy", lambda *_args, **_kwargs: seeds
    )
    direct = runtime.run_direct(
        query="who",
        shared=shared,
        authorization=auth,
        settings=settings,
        deadline=1.0,
    )
    extended = runtime.run_extended(
        prepared=seeds,
        shared=shared,
        authorization=auth,
        settings=settings,
        deadline=1.0,
    )
    assert direct.failure_reason is DirectBranchFailureReason.DIRECT_SEED_INVALID
    assert extended.failure_reason is ExtendedBranchFailureReason.EXTENDED_SEED_INVALID


def test_adaptive_mode_retains_exact_source_authorization_binding():
    snapshot, _config, _seeds, _prepared = _inputs()
    runtime, shared, _auth, settings = _runtime("adaptive", snapshot)
    unrelated = authorization(Policy())
    with pytest.raises(ValueError, match="request binding"):
        runtime.run_direct(
            query="who",
            shared=shared,
            authorization=unrelated,
            settings=settings,
            deadline=1.0,
        )
    with pytest.raises(ValueError, match="request binding"):
        runtime.prepare_extended(
            query="who",
            baseline=object(),
            shared=shared,
            authorization=unrelated,
            settings=settings,
            deadline=1.0,
        )


def test_invalid_policy_in_either_branch_preserves_successful_sibling(monkeypatch):
    from apps.knowledge_graph.retrieval import production_runtime as source

    snapshot, _config, _seeds, prepared = _inputs()
    runtime, shared, auth, settings = _runtime("adaptive", snapshot)
    runtime.scope_loader = lambda **_kwargs: shared.scope
    scheduler = HybridGraphBranchScheduler(runtime, clock=lambda: 100.0)
    request = dict(
        query="who",
        baseline=object(),
        authorization=auth,
        settings=settings,
        deadline=101.0,
    )

    def fail(*_args, **_kwargs):
        raise ValueError("invalid policy")

    monkeypatch.setattr(source, "prepare_direct_with_policy", fail)
    monkeypatch.setattr(
        source, "prepare_extended_with_policy", lambda *_args, **_kwargs: prepared
    )
    direct_failed = scheduler.run(**request)
    assert (
        direct_failed.direct.failure_reason
        is DirectBranchFailureReason.DIRECT_SEED_INVALID
    )
    assert direct_failed.extended.status is BranchStatusV1.SUCCEEDED
    monkeypatch.setattr(
        source, "prepare_direct_with_policy", lambda *_args, **_kwargs: prepared
    )
    monkeypatch.setattr(source, "prepare_extended_with_policy", fail)
    extended_failed = scheduler.run(**request)
    assert extended_failed.direct.status is BranchStatusV1.SUCCEEDED
    assert (
        extended_failed.extended.failure_reason
        is ExtendedBranchFailureReason.EXTENDED_SEED_INVALID
    )
    assert extended_failed.shared_failure_reason is None


def test_extended_policy_preparation_reuses_one_authorized_lookup():
    from apps.knowledge_graph.tests.test_ready_scope import _DOC_A

    lookups = []

    class Repository:
        def load_seed_identities(self, **kwargs):
            lookups.append(kwargs)
            return {10: (key("a"),), 20: (key("b"),)}

    snapshot, _config, _seeds, _prepared = _inputs()
    runtime, shared, auth, settings = _runtime("adaptive", snapshot)
    runtime.projection_repository_factory = Repository
    baseline = SimpleNamespace(
        graph_seeds=(GraphExpansionSeed(10, 1, 0.7), GraphExpansionSeed(20, 2, 0.3)),
        baseline_candidates=(
            SimpleNamespace(pk=10, doc_id=_DOC_A),
            SimpleNamespace(pk=20, doc_id=_DOC_A),
        ),
        vector_chunk_ids=(10, 20),
        trigram_chunk_ids=(10,),
        exact_chunk_ids=(),
    )
    prepared = prepare_extended_with_policy(
        runtime,
        query="connection between a and b",
        baseline=baseline,
        shared=shared,
        authorization=auth,
        settings=settings,
        deadline=1.0,
    )
    assert len(lookups) == 1
    assert {seed.identity_key: seed.mass for seed in prepared.seeds} == {
        key("a"): 0.7,
        key("b"): 0.3,
    }
    assert prepared.support.status == "supported"
    assert prepared.intent == "relational"
