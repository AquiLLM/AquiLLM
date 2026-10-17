"""Packed breadth waves fit the unchanged expanded and wire envelopes."""

import json
from time import monotonic
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.projection.frontier_generation_batch_v2 import attest_all
from apps.knowledge_graph.projection.frontier_index_contracts_v2 import make_record
from apps.knowledge_graph.projection.frontier_preparation_reads_v2 import (
    PreparationReads,
)
from apps.knowledge_graph.projection.frontier_scope_preparation_v2 import (
    ScopePreparation,
)
from apps.knowledge_graph.retrieval.topology.frontier_binary_codec_v2 import (
    decode_binary_response,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
    FrontierGatewayBatchRequestV2,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_service_v2 import (
    read_payload,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_checkpoint_v2 import (
    snapshot,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_codec_v2 import (
    decode_container,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_contracts_v2 import (
    decode_request,
    encode_request,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_owner_v2 import (
    publish,
    replay,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_worker_v2 import (
    read_window,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
    key,
)
from apps.knowledge_graph.tests.test_frontier_packed_paths_v3 import install_packed


def full_directory(generation, ordinal):
    return make_record(
        generation,
        "directory",
        {
            "layout": "lexical-pack-v1",
            "entries": sorted(
                ["evidence/" + key(f"{ordinal}:{i}"), key(f"head:{i}")]
                for i in range(32)
            ),
            "children": [key(f"child:{i}") for i in range(33)],
        },
    )


def test_copy_batch_bound_uses_captured_allowed_predicates():
    fixture = GenerationFixture(1, 1)
    generation = fixture.ready.selected_generations[0].generation_key
    state = PreparationReads(fixture.session.reads)
    state.stage = "copy_entries"
    state.session = SimpleNamespace(
        request=SimpleNamespace(predicate_weights=(("knows", 1.0),))
    )
    pairs = [(generation, key(str(i))) for i in range(160)]
    assert len(state.index_wave(pairs, None).pairs) == 160


@pytest.mark.parametrize(
    "stage, maximum",
    [
        ("scope_entries", 2048),
        ("copy_entries", 32768),
        ("copy_first_heads", 4096),
    ],
)
def test_maximum_writer_entry_shapes_fit_stage_bound_and_batched_payload(
    stage, maximum
):
    fixture = GenerationFixture(1, 1)
    generation = fixture.ready.selected_generations[0].generation_key

    def refs(labels):
        return [[label, "b" * 64, "c" * 64] for label in labels]

    payload = {
        "stream": "scope/" + "a" * 64,
        "order": ["a" * 64],
        "refs": refs(["ArtifactProvenance", "ProjectedDocument"]),
        "links": {},
        "next": None,
        "family": "scope",
    }
    if stage == "copy_entries":
        payload.update(
            stream="copies/" + "a" * 64,
            family="copy",
            next="a" * 64,
            refs=refs(["ProjectedEntity", "AutomaticMembership"]),
            links={
                "relations": {"p" + f"{i:0127d}": "a" * 64 for i in range(128)},
                "mentions": "a" * 64,
            },
        )
    if stage == "copy_first_heads":
        payload.update(
            stream="relations/" + "a" * 64 + "/" + "p" * 128,
            family="relation",
            next="a" * 64,
            order=["f" * 16, "a" * 64, "b" * 64, "c" * 64, "reverse_directed"],
            refs=refs(
                ["ProjectedEntity", "AutomaticMembership"] * 2
                + ["ProjectedRelation", "ProjectedRelationSemantics"]
            ),
            links={"evidence": "a" * 64},
        )
    records = []
    for i in range(160):
        first_order = f"{i:016x}" if stage == "copy_first_heads" else key(str(i))
        records.append(
            make_record(
                generation,
                "entry",
                {**payload, "order": [first_order, *payload["order"][1:]]},
            )
        )
    fixture.driver.index.update((r["index_key"], r) for r in records)
    assert (
        len(
            _canonical(
                {
                    "generation_key": generation,
                    "lookup_key": "a" * 64,
                    "record": records[0],
                }
            )
        )
        + 1
        <= maximum
    )
    state = PreparationReads(fixture.session.reads)
    state.stage = stage
    if stage == "copy_entries":
        state.session = SimpleNamespace(
            request=SimpleNamespace(
                predicate_weights=tuple(
                    (name, 1.0) for name in payload["links"]["relations"]
                )
            )
        )
    pairs = tuple((generation, r["index_key"]) for r in records)
    while (effect := state.index_wave(pairs, None)) is not None:
        frame = read_payload(
            fixture.driver,
            FrontierGatewayBatchRequestV2(
                effect.kind,
                effect.pairs,
                fixture.ready,
                1048576,
            ),
            monotonic() + 5,
            1048576,
            binary=True,
        )
        state.apply_result(tuple(decode_binary_response(frame)["rows"]))
    assert len(fixture.session.reads._index_records) == 160


def test_full_packed_directory_wave_is_split_before_physical_encoding():
    fixture = GenerationFixture(1, 1)
    reads = fixture.session.reads
    generation = fixture.ready.selected_generations[0].generation_key
    records = [full_directory(generation, i) for i in range(136)]
    fixture.driver.index.update((r["index_key"], r) for r in records)
    pairs = tuple((generation, r["index_key"]) for r in records)
    state = PreparationReads(reads)
    state.stage = "copy_directory"
    batch_sizes = []
    while (effect := state.index_wave(pairs, None)) is not None:
        batch_sizes.append(len(effect.pairs))
        frame = read_payload(
            fixture.driver,
            FrontierGatewayBatchRequestV2(
                effect.kind,
                effect.pairs,
                fixture.ready,
                1048576,
            ),
            monotonic() + 5,
            1048576,
            binary=True,
        )
        assert int.from_bytes(frame[8:12], "big") <= 1048576
        state.apply_result(tuple(decode_binary_response(frame)["rows"]))
    assert len(batch_sizes) == 2
    assert sum(batch_sizes) == len(reads._index_records) == 136
    assert fixture.driver.scanned == 136


def test_dense_packed_window_returns_completed_prefix_before_next_large_wave():
    fixture = install_packed(GenerationFixture(4, 8800))
    reads = fixture.session.reads
    reads.reserve_final(multi=True)
    attest_all(reads, fixture.session.selected)
    sources = {}
    for generation, marker in reads.manifests.items():
        root = json.loads(
            fixture.driver.index[marker["frontier_root_key"]]["payload_json"]
        )["payload"]
        sources[generation] = [
            json.loads(fixture.driver.index[child]["payload_json"])["payload"][
                "entries"
            ][0][0].removeprefix("scope/")
            for child in root["children"]
        ]
    state = ScopePreparation(reads, sources)
    completed = []
    for sequence in (1, 2):
        state.next_read()
        operation, checkpoint = snapshot(state)
        grant = fixture.budget.reserve_window()
        body = encode_request(
            fixture.ready,
            operation,
            checkpoint,
            grant,
            sequence,
            fixture.budget.check_active,
        )
        request = decode_request(body)
        before = len(fixture.driver.calls), fixture.driver.scanned
        result = read_window(fixture.driver, request, monotonic() + 5, 1048576)
        frames = decode_container(result, request.request_digest)
        staged, scanned = replay(state, frames, grant)
        fixture.budget.settle_window(grant, queries=len(frames), scanned_rows=scanned)
        publish(state, staged)
        assert len(frames) == len(fixture.driver.calls) - before[0]
        assert scanned == fixture.driver.scanned - before[1]
        completed.append(scanned)
    assert completed[0] == 4
    assert completed[1] > 100
    assert not fixture.budget._uncertain
    assert fixture.budget.queries == len(fixture.driver.calls)
    assert fixture.budget.source_rows == fixture.driver.scanned
