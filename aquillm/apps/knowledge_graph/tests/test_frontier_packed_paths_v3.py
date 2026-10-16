"""Packed roots work through scalar, finite and real gateway checkpoint paths."""

import json

import pytest

from apps.knowledge_graph.projection.frontier_copy_batch_v2 import copy_heads
from apps.knowledge_graph.projection.frontier_index_build_v2 import build_frontier_index
from apps.knowledge_graph.projection.frontier_scope_batch_v2 import scope_rows
from apps.knowledge_graph.projection.frontier_scope_preparation_v2 import (
    ScopePreparation,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_window_owner_v2 import (
    run_preparation,
)
from apps.knowledge_graph.tests.consumed_source_test_support import (
    multi_document_fixture,
)
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
)
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener
from apps.knowledge_graph.tests.test_frontier_packed_directory_v3 import PACKED


def install_packed(fixture):
    fixture.driver.index.clear()
    for bundle in fixture.bundles:
        index = build_frontier_index(bundle, directory_version=PACKED)
        fixture.driver.index.update((r["index_key"], r) for r in index.records)
        fixture.driver.markers[bundle.generation.generation_key].update(index.marker)
    return fixture


@pytest.mark.parametrize("mode", ["scalar", "finite", "window"])
def test_packed_scope_paths_cover_all_documents_and_keep_fresh_authority(
    monkeypatch, mode
):
    fixture = install_packed(GenerationFixture(1, 33))
    reads = fixture.session.reads
    selected = next(iter(fixture.session.selected.values()))
    generation = selected.generation_key
    reads.reserve_final(multi=False)
    reads.attest(selected)
    sources = [
        selected.collection_key,
        *(d.document_key for d in fixture.ready.authorized_documents),
    ]
    transport = WindowOpener(fixture, monkeypatch) if mode == "window" else None
    if mode == "scalar":
        rows = scope_rows(reads, generation, sources)
    else:
        rows = run_preparation(ScopePreparation(reads, {generation: sources}))[
            generation
        ]
    assert sum(label == "ProjectedDocument" for label, _ in rows) == 33
    assert sum(label == "ArtifactProvenance" for label, _ in rows) == 34
    if transport:
        assert any("checkpoint" in json.loads(r.data) for r in transport.requests)
    fixture.session.final_attestation()
    fixture.driver.markers[generation]["graph_checksum"] = "0" * 64
    with pytest.raises(TopologyLoadError):
        reads.attest(selected, final=True)


@pytest.mark.parametrize("mode", ["scalar", "batch", "finite", "window"])
def test_packed_copy_paths_preserve_positive_and_absent_lookup(monkeypatch, mode):
    fixture = install_packed(multi_document_fixture())
    transport = (
        WindowOpener(fixture, monkeypatch, complete_scope=True)
        if mode == "window"
        else None
    )
    fixture.session.initialize()
    bundle = fixture.bundles[0]
    generation = bundle.generation.generation_key
    member = bundle.automatic_memberships[0]
    identity = member.automatic_membership_key or member.entity_key
    pairs = ((generation, identity), (generation, "f" * 64))
    if mode == "scalar":
        assert list(fixture.session.copies(generation, identity))
        assert not list(fixture.session.copies(generation, "f" * 64))
    elif mode == "batch":
        heads = copy_heads(fixture.session, pairs)
        assert heads[pairs[0]] and heads[pairs[1]] is None
    else:
        result = fixture.session.prepare_copies(pairs, relation_heads=True)
        assert result[pairs[0]] and not result[pairs[1]]
    fixture.session.final_attestation()
    assert fixture.budget.queries == len(fixture.driver.calls) <= 64
    assert fixture.budget.source_rows == fixture.driver.scanned <= 5000
    if transport:
        assert any("checkpoint" in json.loads(r.data) for r in transport.requests)


def test_packed_cached_directory_does_not_cache_consumed_raw_proof():
    fixture = install_packed(multi_document_fixture())
    fixture.session.initialize()
    bundle = fixture.bundles[0]
    generation = bundle.generation.generation_key
    member = bundle.automatic_memberships[0]
    identity = member.automatic_membership_key or member.entity_key
    assert list(fixture.session.copies(generation, identity))
    row = fixture.driver.raw[generation, bundle.entities[0].entity_key][0]
    row["record"]["retrieval_utility"] = 0.123
    with pytest.raises(TopologyLoadError):
        list(fixture.session.copies(generation, identity))


def test_missing_packed_child_is_failure_instead_of_absence():
    fixture = install_packed(GenerationFixture(1, 65))
    selected = next(iter(fixture.session.selected.values()))
    fixture.session.reads.attest(selected)
    root = fixture.driver.markers[selected.generation_key]["frontier_root_key"]
    node = json.loads(fixture.driver.index[root]["payload_json"])["payload"]
    del fixture.driver.index[node["children"][0]]
    first = min("scope/" + p.scope_key for p in fixture.bundles[0].artifact_provenance)
    with pytest.raises(TopologyLoadError):
        fixture.session.reads.head(selected.generation_key, first)


def physical_fixture(packed, *, mention_only=False):
    from time import monotonic

    from apps.knowledge_graph.tests.frontier_scope_window_support import (
        MixedScopeFixture,
    )
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    fixture = MixedScopeFixture(mention_only=mention_only)
    if packed:
        install_packed(fixture)
    # Construction is offline; start the unchanged invocation after either build.
    old = fixture.budget
    fixture.budget = type(old)(
        turn_budget=TurnBudget(TurnLimits()),
        deadline=monotonic() + 5.0,
        clock=monotonic,
        **{
            name: getattr(old, name)
            for name in (
                "max_identities",
                "max_edges",
                "max_source_rows",
                "max_queries",
                "max_wire_bytes",
            )
        },
    )
    return fixture


def test_packed_same_five_edge_physical_work_has_material_query_headroom(
    monkeypatch, capsys
):
    from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
        FrontierBudgetExceeded,
    )
    from apps.knowledge_graph.tests import test_consumed_source_physical_v3 as gate

    configure = gate.configure_consumed
    accepted = []

    def capture_configuration(fixture):
        configure(fixture)
        original = fixture.assembly.add_relation

        def capture(page, entry, generation, origin, depth):
            added = original(page, entry, generation, origin, depth)
            if added:
                accepted.append(
                    (
                        generation,
                        entry["order"],
                        depth,
                        [r.evidence_key for r in page.relation_evidence],
                    )
                )
                if len(accepted) == 5:
                    raise FrontierBudgetExceeded("test equal semantic prefix")
            return added

        fixture.assembly.add_relation = capture
        return fixture

    monkeypatch.setattr(gate, "configure_consumed", capture_configuration)
    binary = physical_fixture(False)
    gate.run_physical(binary, monkeypatch, prefix="synthetic-15/294")
    capsys.readouterr()
    expected, accepted = accepted, []
    packed = physical_fixture(True)
    gate.run_physical(packed, monkeypatch, prefix="synthetic-15/294")
    capsys.readouterr()
    assert accepted == expected and len(accepted) == 5
    assert packed.budget.queries <= binary.budget.queries - 20
    assert packed.budget.max_queries - packed.budget.queries >= 20
    assert packed.budget.wire_bytes < binary.budget.wire_bytes
    print(
        "PACKED_EQUAL_PREFIX",
        {
            layout: {
                name: getattr(fixture.budget, name)
                for name in ("queries", "source_rows", "wire_bytes", "edges")
            }
            for layout, fixture in (("binary", binary), ("packed32", packed))
        },
    )


@pytest.mark.parametrize("mention_only", [False, True])
def test_packed_full_physical_gate_retains_fresh_source_snapshot(
    monkeypatch, capsys, mention_only
):
    from apps.knowledge_graph.tests.test_consumed_source_physical_v3 import run_physical

    fixture = physical_fixture(True, mention_only=mention_only)
    _, references = run_physical(fixture, monkeypatch, prefix="synthetic-15/294")
    capsys.readouterr()
    assert len(references) >= 10
    print(
        "PACKED_FULL_GATE",
        {
            "mention_only": mention_only,
            "candidates": len(references),
            **{
                name: getattr(fixture.budget, name)
                for name in ("queries", "source_rows", "wire_bytes", "edges")
            },
        },
    )
