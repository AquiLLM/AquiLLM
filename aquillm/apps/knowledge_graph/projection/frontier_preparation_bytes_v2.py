"""Expanded JSON bounds for the writer-certified finite preparation shapes."""

from .frontier_index_contracts_v2 import MAX_INDEX_RECORD_BYTES, VERSION, canonical

MAX_EXPANDED_BYTES = 1048576
ENVELOPE_BYTES = 128


def index_row_bytes(state, generation):
    # Includes the keyed response wrapper, stored record, escaped payload_json,
    # and a separating comma. Directory names emitted by the builder are at most
    # 73 ASCII bytes (evidence/<sha256>); relations are never directory pivots.
    stage = getattr(state, "stage", "")
    if stage.endswith("directory"):
        version = state.reads.manifests.get(generation, {}).get("frontier_version")
        return 1536 if version == VERSION else 8192
    if stage == "copy_entries" and hasattr(state, "session"):
        # Every copy link must belong to this captured predicate set, including
        # in the reconstructed worker. Account for both layers of JSON quoting.
        return 2048 + sum(
            len(canonical(canonical(name))) + 80
            for name, _ in state.session.request.predicate_weights
        )
    return {
        "scope_entries": 2048,  # Two source references, no links or successor.
        "copy_entries": 32768,  # 128 predicate names of at most 128 ASCII bytes.
        "copy_first_heads": 4096,  # Six source references and one evidence link.
    }.get(stage, 2 * MAX_INDEX_RECORD_BYTES + 1024)


def expanded_bound(state, effect):
    if effect.kind == "index_batch_multi":
        return ENVELOPE_BYTES + sum(
            index_row_bytes(state, generation) for generation, _ in effect.pairs
        )
    return None


def binary_bound(expanded):
    # zlib compressBound plus the twelve-byte KGFZ header; no compression ratio
    # assumption is needed to decide whether another frame fits the wire grant.
    return expanded + (expanded >> 12) + (expanded >> 14) + (expanded >> 25) + 25
