"""Fixed cross-generation authority reads; only immutable indices are cached."""

from copy import deepcopy

from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
    _key,
)

from .frontier_index_contracts_v2 import decode_record
from .frontier_reads_v2 import schema_error


def batches(reads, pairs, *, factor=2, limit=160):
    pending = sorted(set(pairs))
    while pending:
        count = min(limit, reads.budget.available_read_rows() // factor)
        if count < 1:
            raise FrontierBudgetExceeded("frontier physical work exhausted")
        batch, pending = tuple(pending[:count]), pending[count:]
        yield batch


def keyed_rows(rows, pairs, *, raw=False):
    grouped = {pair: [] for pair in pairs}
    fields = {"generation_key", "lookup_key", "record"} | ({"labels"} if raw else set())
    for row in rows:
        if type(row) is not dict or set(row) != fields:
            raise schema_error()
        pair = row["generation_key"], row["lookup_key"]
        if any(type(k) is not str for k in pair) or pair not in grouped:
            raise schema_error()
        grouped[pair].append(row)
    if any(not values or (not raw and len(values) != 1) for values in grouped.values()):
        raise schema_error()
    return grouped


def attest_all(reads, selected, *, final=False):
    pairs = tuple(sorted((g, g) for g in selected))
    # Ready bundles cap generations at 128; this is exactly one physical read.
    physical = reads.read_batch("manifest_batch", pairs, final=final)
    rows = keyed_rows(physical, pairs)
    indexed, markers = {}, {}
    for (generation, _), values in rows.items():
        marker = values[0]["record"]
        indexed[generation] = reads.validate_marker(
            selected[generation], marker, final=final, store=False
        )
        markers[generation] = marker
    reads._check_index_deadline()
    if not final:
        reads.reserve_final_wire(physical, multi=True)
        reads.manifests.update(markers)
    return indexed


def indices_all(reads, pairs):
    reads._check_index_deadline()
    pairs = tuple(sorted(set(pairs)))
    selected = {s.generation_key for s in reads.ready.selected_generations}
    for generation, key in pairs:
        _key(generation)
        _key(key)
        if generation not in selected:
            raise schema_error()
    missing = [pair for pair in pairs if pair not in reads._index_records]
    decoded = {}
    for batch in batches(reads, missing):
        rows = keyed_rows(reads.read_batch("index_batch_multi", batch), batch)
        for (generation, key), values in rows.items():
            try:
                value = decode_record(
                    values[0]["record"], expected_key=key, generation=generation
                )
                if value[0] not in {"directory", "entry"}:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise schema_error() from None
            decoded[generation, key] = value
    reads._check_index_deadline()
    reads._index_records.update(decoded)
    return {pair: deepcopy(reads._index_records[pair]) for pair in pairs}


def hydrate_all(reads, expected):
    pairs = {(generation, key) for generation, _, key in expected}
    result = {generation: {} for generation, _ in pairs}
    for batch in batches(reads, pairs, factor=10, limit=32):
        grouped = keyed_rows(
            reads.read_batch("raw_batch_multi", batch), batch, raw=True
        )
        for generation in sorted({g for g, _ in batch}):
            keys = {k for g, k in batch if g == generation}
            refs = {
                (label, key): checksum
                for (g, label, key), checksum in expected.items()
                if g == generation and key in keys
            }
            rows = tuple(
                {k: v for k, v in row.items() if k != "generation_key"}
                for pair, values in grouped.items()
                if pair[0] == generation
                for row in values
            )
            result[generation].update(reads.validate_raw(generation, refs, keys, rows))
    return result
