"""Atomic admission of bounded, authenticated immutable index records."""

from copy import deepcopy

from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import _key

from .frontier_index_contracts_v2 import decode_record
from .frontier_reads_v2 import schema_error


def read_indices(reads, generation, keys):
    reads._check_index_deadline()
    _key(generation)
    if generation not in {s.generation_key for s in reads.ready.selected_generations}:
        raise schema_error()
    if type(keys) is not tuple or not 1 <= len(keys) <= 32:
        raise schema_error()
    for key in keys:
        _key(key)
    if len(set(keys)) != len(keys):
        raise schema_error()
    missing = tuple(sorted(k for k in keys if (generation, k) not in reads._index_records))
    decoded = {}
    if missing:
        rows = reads.read("index_batch", generation, missing)
        for row in rows:
            if type(row) is not dict or set(row) != {"lookup_key", "record"}:
                raise schema_error()
            key = row["lookup_key"]
            if type(key) is not str or key not in missing or key in decoded:
                raise schema_error()
            try:
                value = decode_record(row["record"], expected_key=key, generation=generation)
                if value[0] not in {"directory", "entry"}:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise schema_error() from None
            decoded[key] = value
        if set(decoded) != set(missing):
            raise schema_error()
    # Failed batches never leave an authenticated prefix reusable by a later call.
    reads._check_index_deadline()
    reads._index_records.update(((generation, k), v) for k, v in decoded.items())
    result = {k: deepcopy(reads._index_records[generation, k]) for k in keys}
    reads._check_index_deadline()
    return result
