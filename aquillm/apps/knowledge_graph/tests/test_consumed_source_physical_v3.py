"""Original full-scope client/ASGI/worker/codec/owner physical gate, not live corpus."""

import pytest

from apps.knowledge_graph.projection.frontier_snapshot_v2 import frontier_config
from apps.knowledge_graph.retrieval.projected_relation_ranking_v2 import (
    rank_relations_v2,
)
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.retrieval.topology.frontier_acquisition_v2 import (
    acquire_frontier,
)
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.tests.consumed_source_test_support import configure_consumed
from apps.knowledge_graph.tests.frontier_scope_memgraph_support import (
    live_scope as _live_scope,
)
from apps.knowledge_graph.tests.frontier_scope_window_support import MixedScopeFixture
from apps.knowledge_graph.tests.test_frontier_copy_feasibility_v2 import GatewayOpener


@pytest.fixture(params=[False, True], ids=["relations", "mentions"])
def live_scope(request, monkeypatch):
    from apps.knowledge_graph.tests import frontier_scope_memgraph_support as support

    monkeypatch.setattr(
        support,
        "MixedScopeFixture",
        lambda: MixedScopeFixture(mention_only=request.param),
    )
    yield from _live_scope.__wrapped__()


def run_physical(fixture, monkeypatch, *, prefix):
    fixture = configure_consumed(fixture)
    if prefix == "synthetic-15/294":
        for rows in [1, 1, 1, 7, 1, 264, 1, 7, 1, 1, 1, 1, 1, 1, 5]:
            fixture.budget.reserve_read(rows)
            fixture.budget.charge_read(reserved_rows=rows, scanned_rows=rows)
    initial_prefix = fixture.budget.queries, fixture.budget.source_rows
    transport = GatewayOpener(fixture, monkeypatch, windows=True, complete_scope=True)
    seeds = tuple(
        ProjectedSeedV1(b.entity_mentions[0].entity_key, 1 / 16)
        for b in sorted(fixture.bundles, key=lambda b: b.generation.generation_key)[:16]
    )
    unsorted_seeds = seeds
    seeds = tuple(sorted(seeds, key=lambda s: s.identity_key))
    assert len(seeds) == 16 and set(seeds) == set(unsorted_seeds)
    assert all(seed.mass == 1 / 16 for seed in seeds)
    first_candidate = []
    original = fixture.assembly.add_relation

    def retain(*args):
        result = original(*args)
        if result and not first_candidate:
            first_candidate.append(counts())
        return result

    fixture.assembly.add_relation = retain
    original_mention = fixture.assembly.add_mention

    def retain_mention(*args):
        result = original_mention(*args)
        if not first_candidate:
            first_candidate.append(counts())
        return result

    fixture.assembly.add_mention = retain_mention

    def counts():
        return (
            fixture.budget.queries,
            fixture.budget.source_rows,
            fixture.budget.wire_bytes,
        )

    fixture.session.initialize()
    initialized = counts()
    assert (
        sum(
            label == "ArtifactProvenance"
            for rows in fixture.session.scope_rows.values()
            for label, _ in rows
        )
        == 31
    )
    fixture.driver.phase = "acquisition"
    stop = None
    try:
        acquire_frontier(
            fixture.session, fixture.assembly, seeds, fixture.caps, fixture.budget
        )
    except FrontierBudgetExceeded:
        stop = "frontier_budget"
    fixture.driver.phase = "final"
    fixture.session.final_attestation()
    snapshot, references = fixture.assembly.build(fixture.request)
    ranked = rank_relations_v2(
        snapshot=snapshot,
        seeds=seeds,
        request=fixture.request,
        config=frontier_config(
            fixture.caps, fixture.ready.selected_generations[0].resolver_version
        ),
        frontier_context=fixture.assembly.context,
        deadline_check=fixture.budget.check_active,
    )
    assert ranked.scores and references and first_candidate
    assert bool(snapshot.mentions) == (not fixture.bundles[0].relations)
    assert len(snapshot.allowed_scope.document_keys) == 530
    assert len(snapshot.allowed_scope.collection_keys) == 31
    assert (
        snapshot.allowed_scope.scope_version_signature == fixture.group.group_checksum
    )
    assert {s.identity_key for s in seeds} <= fixture.budget._identities
    assert fixture.budget.queries == initial_prefix[0] + len(fixture.driver.calls) <= 64
    assert (
        fixture.budget.source_rows == initial_prefix[1] + fixture.driver.scanned <= 5000
    )
    assert (
        fixture.budget.wire_bytes == sum(f["wire"] for f in transport.frames) <= 1048576
    )
    assert not fixture.budget._uncertain and not fixture.budget._authority_reservations
    assert fixture.budget._window is None and fixture.budget._reserved_rows == 0
    assert fixture.driver.transcript[-1]["phase"] == "final"
    assert fixture.driver.transcript[-1]["rows"] == 31
    print(
        {
            "prefix": prefix,
            "prefix_counts": initial_prefix,
            "initialized": initialized,
            "first_candidate": first_candidate[0],
            "final": counts(),
            "identities": fixture.budget.identities,
            "candidates": len(references),
            "mentions": len(snapshot.mentions),
            "stop": stop,
            "transcript": fixture.driver.transcript,
            "frames": transport.frames,
        }
    )
    return fixture, references


@pytest.mark.parametrize("mention_only", [False, True])
def test_full_31_530_all16_emulator_gateway_proof_and_snapshot(
    monkeypatch, mention_only
):
    run_physical(
        MixedScopeFixture(mention_only=mention_only),
        monkeypatch,
        prefix="synthetic-15/294",
    )


def test_full_31_530_all16_actual_memgraph_gateway_proof_and_snapshot(
    live_scope, monkeypatch
):
    fixture, _ = live_scope
    run_physical(fixture, monkeypatch, prefix="synthetic-15/294")
