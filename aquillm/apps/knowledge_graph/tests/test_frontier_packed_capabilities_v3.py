"""Captured recipes and protocol fingerprints fail closed across rollout."""

import json

import pytest

from apps.knowledge_graph.projection.frontier_generation_batch_v2 import attest_all
from apps.knowledge_graph.projection.frontier_index_build_v2 import build_frontier_index
from apps.knowledge_graph.projection.frontier_preparation_reads_v2 import (
    PreparationReads,
)
from apps.knowledge_graph.projection.frontier_scope_complete_v3 import (
    ScopeCompletePreparation,
)
from apps.knowledge_graph.projection.frontier_scope_preparation_v2 import (
    ScopePreparation,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_window_checkpoint_v2 import (
    snapshot,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_owner_v2 import (
    run_preparation,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
)
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener
from apps.knowledge_graph.tests.test_frontier_packed_byte_bounds_v3 import (
    full_directory,
)
from apps.knowledge_graph.tests.test_frontier_packed_paths_v3 import install_packed


def test_mixed_attested_recipes_share_one_owner_and_worker_ledger(monkeypatch):
    fixture = install_packed(GenerationFixture(2, 64))
    bundle = fixture.bundles[0]
    binary = build_frontier_index(bundle)
    fixture.driver.index.update((r["index_key"], r) for r in binary.records)
    fixture.driver.markers[bundle.generation.generation_key].update(binary.marker)
    transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
    fixture.session.initialize()
    fixture.session.final_attestation()
    assert (
        sum(
            label == "ProjectedDocument"
            for rows in fixture.session.scope_rows.values()
            for label, _ in rows
        )
        == 64
    )
    assert fixture.budget.queries == len(fixture.driver.calls)
    assert fixture.budget.source_rows == fixture.driver.scanned
    assert not fixture.budget._uncertain
    assert any(
        request.full_url.endswith("/scope-window") for request in transport.requests
    )


def test_checkpoint_carries_attested_directory_recipe():
    fixture = install_packed(GenerationFixture(1, 1))
    reads = fixture.session.reads
    attest_all(reads, fixture.session.selected)
    generation, selected = next(iter(fixture.session.selected.items()))
    state = ScopePreparation(reads, {generation: [selected.collection_key]})
    _, checkpoint = snapshot(state)
    assert checkpoint["directory_versions"] == [
        [0, "canonical-frontier-index-v3-packed32"]
    ]


def test_captured_binary_recipe_cannot_admit_packed_directory():
    fixture = GenerationFixture(1, 1)
    reads = fixture.session.reads
    attest_all(reads, fixture.session.selected)
    generation = fixture.ready.selected_generations[0].generation_key
    row = full_directory(generation, 1)
    state = PreparationReads(reads)
    state.stage = "copy_directory"
    state.index_wave(((generation, row["index_key"]),), None)
    with pytest.raises(TopologyLoadError):
        state.apply_result(
            (
                {
                    "generation_key": generation,
                    "lookup_key": row["index_key"],
                    "record": row,
                },
            )
        )
    assert not reads._index_records


@pytest.mark.parametrize("mutation", ["unknown_recipe", "old_checksum"])
@pytest.mark.parametrize("complete_scope", [False, True])
def test_window_recipe_incompatibility_rejected_before_any_physical_read(
    monkeypatch, mutation, complete_scope
):
    fixture = install_packed(GenerationFixture(1, 1))
    reads = fixture.session.reads
    attest_all(reads, fixture.session.selected)
    generation, selected = next(iter(fixture.session.selected.items()))
    state_type = ScopeCompletePreparation if complete_scope else ScopePreparation
    state = state_type(reads, {generation: [selected.collection_key]})
    transport = WindowOpener(fixture, monkeypatch, complete_scope=complete_scope)
    original = transport.open

    def incompatible(request, timeout):
        if mutation == "unknown_recipe":
            value = json.loads(request.data)
            value["checkpoint"]["directory_versions"][0][1] = "unsupported-recipe"
            request.data = _canonical(value)
        else:
            from hashlib import sha256

            old = (
                b"finite-scope-copy-v2:ready-checkpoint-grant-sequence:pair-mask25600:"
                b"KGFW0001:sha256:frames8:raw4:rows320:expanded1048576:wire1048576:"
                b"request4194304:owner-replay"
            )
            if complete_scope:
                old = (
                    b"scope-complete-window-v3:mode-window-or-single:"
                    b"ready-operation-checkpoint-grant-sequence:scope-only:KGFS0001:"
                    b"sha256:index160-limit2:raw128-limit10-total320-reject320:"
                    b"raw-reserve320:frames8:raw4:rows320:expanded1048576:"
                    b"wire1048576:request4194304:owner-replay:"
                    b"single-predispatch-checkpoint-only"
                )
            request.add_header("X-Topology-Schema-Checksum", sha256(old).hexdigest())
        return original(request, timeout)

    transport.open = incompatible
    before = len(fixture.driver.calls)
    with pytest.raises(TopologyLoadError):
        run_preparation(state)
    assert len(fixture.driver.calls) == before
    assert len(transport.requests) == 1
    assert transport.responses[0].status == 400
