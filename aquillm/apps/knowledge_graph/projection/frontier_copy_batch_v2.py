"""Authenticate every competing copy stream before admitting a depth's edges."""

from .frontier_copy_preparation_v2 import CopyPreparation  # noqa: F401
from .frontier_directory_proof_v2 import directory_step
from .frontier_generation_batch_v2 import indices_all
from .frontier_reads_v2 import schema_error


def copy_heads(session, pairs):
    reads = session.reads
    pending = {
        pair: (reads.manifests[pair[0]]["frontier_root_key"], set(), "", "\uffff")
        for pair in pairs
        if session.indexed[pair[0]]
    }
    heads = {}
    while pending:
        missing = set()
        for (generation, identity), (key, seen, low, high) in tuple(pending.items()):
            while key is not None:
                reads._check_index_deadline()
                if key in seen or len(seen) > 32:
                    raise schema_error()
                if (generation, key) not in reads._index_records:
                    missing.add((generation, key))
                    break
                seen.add(key)
                found, key, low, high = directory_step(
                    reads.index(generation, key), "copies/" + identity, low, high
                )
                if found:
                    heads[generation, identity] = key
                    break
            pair = generation, identity
            if key is None or pair in heads:
                heads.setdefault(pair, None)
                del pending[pair]
            else:
                pending[pair] = key, seen, low, high
        indices_all(reads, missing)
    return heads


def copy_entries(session, heads):
    reads = session.reads
    entries = {pair: [] for pair in heads}
    pending = {
        pair: (key, None, set()) for pair, key in heads.items() if key is not None
    }
    while pending:
        indices_all(reads, ((g, state[0]) for (g, _), state in pending.items()))
        following = {}
        for (generation, identity), (key, previous, seen) in pending.items():
            reads._check_index_deadline()
            if key in seen:
                raise schema_error()
            seen.add(key)
            entry = reads.entry(
                generation, key, stream="copies/" + identity, previous=previous
            )
            session.validate_copy_entry(entry)
            entries[generation, identity].append(entry)
            if entry["next"] is not None:
                following[generation, identity] = (
                    entry["next"],
                    tuple(entry["order"]),
                    seen,
                )
        pending = following
    return entries


def prepare_copies(session, pairs, *, relation_heads):
    """Return fresh per-operation raw subsets, never an authority cache."""
    from apps.knowledge_graph.retrieval.topology.frontier_window_owner_v2 import (
        run_preparation,
    )

    return run_preparation(
        CopyPreparation(session, pairs, relation_heads=relation_heads)
    )
