"""Both derived directory recipes preserve authenticated lexical seeks."""

import json
from copy import deepcopy

import pytest

from apps.knowledge_graph.projection.frontier_directory_proof_v2 import directory_step
from apps.knowledge_graph.projection.frontier_index_build_v2 import build_frontier_index
from apps.knowledge_graph.projection.frontier_index_contracts_v2 import (
    VERSION,
    decode_record,
    digest,
    make_record,
    marker_present,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.tests.test_projection_records import _bundle

PACKED = "canonical-frontier-index-v3-packed32"


def seek(index, generation, name):
    rows = {r["index_key"]: r for r in index.records}
    key, low, high, seen = index.marker["frontier_root_key"], "", "\uffff", set()
    while key is not None:
        assert key not in seen and len(seen) <= 32
        seen.add(key)
        row = decode_record(rows[key], expected_key=key, generation=generation)
        found, key, low, high = directory_step(row, name, low, high)
        if found:
            return key
    return None


def test_legacy_default_remains_byte_exact():
    index = build_frontier_index(_bundle())
    assert index.marker["frontier_version"] == VERSION
    assert index.marker["frontier_checksum"] == (
        "a14eb4a80af4c4cb193e25c75ef35280c0cbadb685c52ef79d6aad6131b98563"
    )
    assert digest(index.records) == (
        "c59907f090edade76092dc6400ba28bc8896bc1b00ce6363798dc1c2233ee3e2"
    )


def test_packed_build_preserves_entry_bytes_heads_and_authenticated_absence():
    bundle = _bundle()
    binary = build_frontier_index(bundle)
    packed = build_frontier_index(bundle, directory_version=PACKED)
    assert packed == build_frontier_index(bundle, directory_version=PACKED)
    assert packed.marker["frontier_version"] == PACKED
    assert marker_present(packed.marker)
    assert all(row["version"] == VERSION for row in packed.records)
    assert [r for r in binary.records if r["kind"] == "entry"] == [
        r for r in packed.records if r["kind"] == "entry"
    ]
    for row in binary.records:
        if row["kind"] == "directory":
            node = json.loads(row["payload_json"])["payload"]
            assert (
                seek(packed, bundle.generation.generation_key, node["name"])
                == node["head"]
            )
    assert seek(packed, bundle.generation.generation_key, "copies/" + "f" * 64) is None
    with pytest.raises(ValueError):
        build_frontier_index(bundle, directory_version="unknown")


@pytest.mark.parametrize(
    "mutation",
    ["order", "duplicate", "tag", "head", "child", "count", "extra", "bounds"],
)
def test_packed_transition_refuses_malformed_authenticated_payload(mutation):
    node = {
        "layout": "lexical-pack-v1",
        "entries": [["copies/a", "a" * 64], ["copies/b", "b" * 64]],
        "children": [None, None, None],
    }
    if mutation == "order":
        node["entries"].reverse()
    elif mutation == "duplicate":
        node["entries"][1] = node["entries"][0]
    elif mutation == "tag":
        node["layout"] = "unknown"
    elif mutation == "head":
        node["entries"][0][1] = None
    elif mutation == "child":
        node["children"][2] = "bad"
    elif mutation == "count":
        node["children"].pop()
    elif mutation == "extra":
        node["extra"] = 1
    with pytest.raises(TopologyLoadError):
        directory_step(
            ("directory", node),
            "copies/a",
            "copies/b" if mutation == "bounds" else "",
            "\uffff",
        )


def test_packed_content_link_and_generation_are_not_optional():
    row = make_record(
        "a" * 64,
        "directory",
        {
            "layout": "lexical-pack-v1",
            "entries": [["copies/a", "b" * 64]],
            "children": [None, None],
        },
    )
    damaged = deepcopy(row)
    damaged["payload_json"] += " "
    with pytest.raises(ValueError):
        decode_record(damaged, expected_key=row["index_key"], generation="a" * 64)
    with pytest.raises(ValueError):
        decode_record(row, expected_key=row["index_key"], generation="c" * 64)


@pytest.mark.parametrize("name", ["copies/0", "copies/z", None, 1])
def test_packed_lookup_target_must_be_inside_exact_checkpoint_range(name):
    node = {
        "layout": "lexical-pack-v1",
        "entries": [["copies/b", "a" * 64]],
        "children": [None, None],
    }
    with pytest.raises(TopologyLoadError):
        directory_step(("directory", node), name, "copies/a", "copies/c")


@pytest.mark.parametrize("count", [1, 32, 33, 4096, 60001])
def test_packed_directory_boundary_and_long_shared_prefix(count):
    from apps.knowledge_graph.projection.frontier_packed_directory_v3 import (
        packed_directory,
    )

    records = []

    def put(kind, payload):
        row = make_record("a" * 64, kind, payload)
        records.append(row)
        return row["index_key"]

    items = [("copies/" + "s" * 240 + f"{i:08d}", f"{i:064x}") for i in range(count)]
    root = packed_directory(items, put)
    index = type(
        "Index", (), {"records": records, "marker": {"frontier_root_key": root}}
    )()
    for i in sorted({0, count // 2, count - 1}):
        assert seek(index, "a" * 64, items[i][0]) == items[i][1]
    assert seek(index, "a" * 64, "copies/" + "t" * 64) is None
    assert max(len(r["payload_json"].encode()) for r in records) <= 65536
