"""Owner checkpoints for two fixed preparations; cached bodies never cross."""

from types import SimpleNamespace

from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2
from apps.knowledge_graph.projection.frontier_copy_preparation_v2 import CopyPreparation
from apps.knowledge_graph.projection.frontier_scope_preparation_v2 import (
    ScopePreparation,
)

from .frontier_window_mask_v2 import decode_mask, encode_mask


def snapshot(state):
    reads = state.reads
    generations = [g.generation_key for g in reads.ready.selected_generations]
    slots = {g: i for i, g in enumerate(generations)}
    copying = isinstance(state, CopyPreparation)
    identities = sorted({identity for _, identity in state.pairs}) if copying else []

    def target(g, name):
        return [
            slots[g],
            identities.index(name) if copying else name.removeprefix("scope/"),
        ]

    operation = {
        "kind": "copy" if copying else "scope",
        "roots": [
            [slots[g], marker["frontier_root_key"]]
            for g, marker in reads.manifests.items()
            if marker.get("frontier_root_key")
        ],
        "request_digest": getattr(reads, "request_digest", reads.ready.bundle_checksum),
    }
    if copying:
        operation.update(
            identities=identities,
            pair_mask=encode_mask(
                [target(*pair) for pair in state.pairs],
                len(generations),
                len(identities),
            ),
            weights=[list(p) for p in state.session.request.predicate_weights],
            relation_heads=state.relation_heads,
        )
    checkpoint = {
        "stage": state.stage,
        "guard": [
            [slots[g], k]
            for g, k in sorted(set(reads._index_records) | set(state.staged))
        ],
        "wave": state.wave is not None,
    }
    if state.stage.endswith("directory"):
        checkpoint["paths"] = [
            target(g, name) + [key, low, high, prior + len(seen)]
            for (g, name), (key, seen, low, high, prior) in state.pending.items()
        ]
        checkpoint["heads"] = [
            target(g, name) + [key] for (g, name), key in state.heads.items()
        ]
    if state.stage == "scope_entries":
        checkpoint["heads"] = [
            target(g, name) + [key] for (g, name), key in state.heads.items()
        ]
    if state.stage == "copy_entries":
        checkpoint["paths"] = [
            target(g, name) + [key, list(previous) if previous is not None else None]
            for (g, name), (key, previous, _) in state.pending.items()
        ]
    if state.stage in {"copy_entries", "copy_raw"}:
        checkpoint["entries"] = [
            target(g, name) + [entry]
            for (g, name), entries in state.entries.items()
            for entry in entries
        ]
    if state.stage.endswith("raw"):
        checkpoint["expected"] = [
            [slots[g], label, key, checksum]
            for (g, label, key), checksum in state.expected.items()
        ]
        checkpoint["raw_pending"] = [[slots[g], key] for g, key in state.raw_pending]
        if copying:
            checkpoint["earlier_raw"] = any(state.fresh.values())
    if state.stage == "copy_first_heads":
        checkpoint["first"] = [
            [slots[g], head, stream] for g, head, stream in state.first
        ]
    return operation, checkpoint


class CopyContext:
    validate_copy_entry = FrontierSessionV2.validate_copy_entry
    validate_copy_rows = FrontierSessionV2.validate_copy_rows

    def __init__(self, reads, weights, roots):
        self.reads, self.ready = reads, reads.ready
        self.selected = {g.generation_key: g for g in self.ready.selected_generations}
        self.indexed = {g: g in roots for g in self.selected}
        self.request = SimpleNamespace(
            predicate_weights=tuple(tuple(w) for w in weights)
        )


def restore(reads, operation, checkpoint):
    from .frontier_window_validation_v2 import validate_checkpoint

    validate_checkpoint(reads.ready, operation, checkpoint)
    generations = [g.generation_key for g in reads.ready.selected_generations]
    roots = {generations[i]: key for i, key in operation["roots"]}
    reads.manifests = {g: {"frontier_root_key": key} for g, key in roots.items()}
    copying = operation["kind"] == "copy"

    def target(slot, name):
        return generations[slot], operation["identities"][
            name
        ] if copying else "scope/" + name

    if copying:
        context = CopyContext(reads, operation["weights"], roots)
        state = CopyPreparation(
            context,
            [
                target(*p)
                for p in decode_mask(
                    operation["pair_mask"],
                    len(generations),
                    len(operation["identities"]),
                )
            ],
            relation_heads=operation["relation_heads"],
        )
    else:
        state = ScopePreparation(reads, {})
    state.stage = checkpoint["stage"]
    state.guard = {(generations[i], key) for i, key in checkpoint["guard"]}
    state.pending = {}
    if state.stage.endswith("directory"):
        for i, name, key, low, high, depth in checkpoint["paths"]:
            visited = {key} if not copying and checkpoint["wave"] else set()
            state.pending[target(i, name)] = (
                key,
                visited,
                low,
                high,
                depth - len(visited),
            )
        state.heads = {target(i, name): key for i, name, key in checkpoint["heads"]}
    if state.stage == "scope_entries":
        state.heads = {target(i, name): key for i, name, key in checkpoint["heads"]}
    if state.stage == "copy_entries":
        state.pending = {
            target(i, name): (
                key,
                tuple(previous) if previous is not None else None,
                set(),
            )
            for i, name, key, previous in checkpoint["paths"]
        }
    if state.stage in {"copy_entries", "copy_raw"}:
        state.entries = {pair: [] for pair in state.pairs}
        for i, name, entry in checkpoint["entries"]:
            state.entries[target(i, name)].append(entry)
    if state.stage.endswith("raw"):
        state.expected = {
            (generations[i], label, key): checksum
            for i, label, key, checksum in checkpoint["expected"]
        }
        state.raw_pending = [
            (generations[i], key) for i, key in checkpoint["raw_pending"]
        ]
        state.earlier_raw = checkpoint.get("earlier_raw", False)
    if state.stage == "copy_first_heads":
        state.first = [
            (generations[i], key, stream) for i, key, stream in checkpoint["first"]
        ]
    if checkpoint["wave"]:
        if state.stage == "scope_entries":
            targets = ((g, head) for (g, _), head in state.heads.items())
        elif state.stage == "copy_first_heads":
            targets = ((g, head) for g, head, _ in state.first)
        else:
            targets = ((g, s[0]) for (g, _), s in state.pending.items())
        targets = tuple(sorted(set(targets)))
        state.wave = targets, [p for p in targets if p not in state.guard]
    return state
