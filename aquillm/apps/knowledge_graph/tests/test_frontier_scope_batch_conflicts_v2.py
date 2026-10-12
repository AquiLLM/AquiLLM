"""Conflicting authenticated scope references cannot disappear during aggregation."""

from copy import deepcopy

import pytest

from apps.knowledge_graph.projection.frontier_cypher_v2 import RAW_LOOKUP
from apps.knowledge_graph.projection.frontier_index_contracts_v2 import (
    VERSION, decode_record, digest, make_record,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.tests.frontier_scope_batch_support import ScopeBatchFixture


def _replace_scope_entry(fixture, target, extra_ref):
    """Synthetic corrupt-but-authenticated index; real raw DTOs are untouched."""
    generation = fixture.bundle.generation.generation_key
    marker = fixture.driver.markers[generation]
    original, rebuilt = fixture.driver.index, {}

    def rebuild(key):
        if key is None:
            return None
        kind, value = decode_record(original[key], expected_key=key, generation=generation)
        value = deepcopy(value)
        if kind == "directory":
            for field in ("left", "right", "head"):
                value[field] = rebuild(value[field])
        elif kind == "entry" and value["stream"] == target:
            value["refs"].append(extra_ref)
        row = make_record(generation, kind, value)
        rebuilt[row["index_key"]] = row
        return row["index_key"]

    root = rebuild(marker["frontier_root_key"])
    fixture.driver.index.update(rebuilt)
    records = tuple(fixture.driver.index[k] for k in sorted(fixture.driver.index))
    checksum = digest({"version": VERSION, "graph_checksum": marker["graph_checksum"],
                       "records": records, "root": root})
    marker.update(frontier_record_count=len(records),
                  frontier_root_key=root, frontier_root_checksum=root,
                  frontier_checksum=checksum, validated_frontier_checksum=checksum)


@pytest.mark.parametrize("documents", [2, 32])
@pytest.mark.parametrize("bad_first", [False, True])
def test_conflicting_refs_are_rejected_before_raw_grouping_or_last_write_wins(documents, bad_first):
    fixture = ScopeBatchFixture(documents)
    ready = fixture.group.scope.ready
    generation = fixture.bundle.generation.generation_key
    sources = [fixture.bundle.generation.collection_key,
               *(d.document_key for d in ready.authorized_documents)]
    first, last = "scope/" + sources[0], "scope/" + sources[-1]
    entries = {}
    for key, row in fixture.driver.index.items():
        kind, value = decode_record(row, expected_key=key, generation=generation)
        if kind == "entry" and value["stream"] in {first, last}:
            entries[value["stream"]] = value
    victim, target = (last, first) if bad_first else (first, last)
    duplicate = deepcopy(entries[victim]["refs"][0])
    duplicate[2] = "0" * 64
    _replace_scope_entry(fixture, target, duplicate)
    raw_before = deepcopy(fixture.driver.raw)
    with pytest.raises(TopologyLoadError):
        fixture.session.initialize()
    assert not any(query == RAW_LOOKUP for query, _ in fixture.driver.calls)
    assert fixture.driver.raw == raw_before
    assert fixture.session.scope_rows == {}
    assert fixture.budget.queries == len(fixture.driver.calls)
    assert fixture.budget.source_rows == fixture.driver.scanned
    assert fixture.budget._reserved_rows == 0 and not fixture.budget._uncertain
