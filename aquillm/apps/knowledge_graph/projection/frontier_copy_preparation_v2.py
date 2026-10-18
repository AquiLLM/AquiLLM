"""Finite cache-first copy chains and fresh raw-dependent first-head reads."""

from .frontier_directory_proof_v2 import directory_step
from .frontier_preparation_reads_v2 import OwnerBoundary, PreparationReads
from .frontier_reads_v2 import schema_error


class CopyPreparation(PreparationReads):
    def __init__(self, session, pairs, *, relation_heads):
        super().__init__(session.reads)
        self.session, self.pairs, self.relation_heads = (
            session,
            tuple(pairs),
            relation_heads,
        )
        self.stage = "copy_directory"
        self.pending = {
            pair: (
                self.reads.manifests[pair[0]]["frontier_root_key"],
                set(),
                "",
                "\uffff",
                0,
            )
            for pair in pairs
            if session.indexed[pair[0]]
        }
        self.heads, self.entries, self.first = {}, {}, []
        self.prepared = {pair: [] for pair in pairs}

    def next_read(self, allowance=None):
        while self.result is None:
            self.reads._check_index_deadline()
            if self.stage == "copy_directory":
                if self.wave is not None:
                    effect = self.index_wave((), allowance)
                    if effect:
                        return effect
                missing = set()
                for (g, identity), (key, seen, low, high, prior) in tuple(
                    self.pending.items()
                ):
                    while key is not None:
                        self.reads._check_index_deadline()
                        if key in seen or prior + len(seen) > 32:
                            raise schema_error()
                        if (g, key) not in self.reads._index_records:
                            if (g, key) in self.guard:
                                raise OwnerBoundary
                            missing.add((g, key))
                            break
                        seen.add(key)
                        found, key, low, high = directory_step(
                            self.reads.index(g, key), "copies/" + identity, low, high
                        )
                        if found:
                            self.heads[g, identity] = key
                            break
                    pair = g, identity
                    if key is None or pair in self.heads:
                        self.heads.setdefault(pair, None)
                        del self.pending[pair]
                    else:
                        self.pending[pair] = key, seen, low, high, prior
                if missing:
                    return self.index_wave(missing, allowance)
                self.entries = {pair: [] for pair in self.heads}
                self.pending = {
                    pair: (key, None, set())
                    for pair, key in self.heads.items()
                    if key is not None
                }
                self.stage = "copy_entries"
            elif self.stage == "copy_entries":
                if not self.pending:
                    for (g, _), entries in self.entries.items():
                        for entry in entries:
                            self.add_refs(g, entry["refs"])
                    self.stage = "copy_raw"
                    continue
                effect = self.index_wave(
                    ((g, s[0]) for (g, _), s in self.pending.items()), allowance
                )
                if effect:
                    return effect
                following = {}
                for (g, identity), (key, previous, seen) in self.pending.items():
                    if key in seen:
                        raise schema_error()
                    seen.add(key)
                    entry = self.reads.entry(
                        g, key, stream="copies/" + identity, previous=previous
                    )
                    self.session.validate_copy_entry(entry)
                    self.entries[g, identity].append(entry)
                    if entry["next"] is not None:
                        following[g, identity] = (
                            entry["next"],
                            tuple(entry["order"]),
                            seen,
                        )
                self.pending = following
            elif self.stage == "copy_raw":
                effect = self.raw_read(allowance)
                if effect:
                    return effect
                if self.earlier_raw:
                    raise OwnerBoundary
                for (g, identity), entries in self.entries.items():
                    for entry in entries:
                        rows = {
                            (label, key): self.fresh[g][label, key]
                            for label, key, _ in entry["refs"]
                        }
                        self.session.validate_copy_rows(g, identity, entry, rows)
                        self.prepared[g, identity].append((entry, rows))
                        if self.relation_heads:
                            for predicate, head in sorted(
                                entry["links"]["relations"].items()
                            ):
                                if head is not None:
                                    self.first.append(
                                        (
                                            g,
                                            head,
                                            "relations/"
                                            + entry["order"][0]
                                            + "/"
                                            + predicate,
                                        )
                                    )
                self.stage = "copy_first_heads"
            elif self.stage == "copy_first_heads":
                effect = self.index_wave(
                    ((g, head) for g, head, _ in self.first), allowance
                )
                if effect:
                    return effect
                for g, head, stream in self.first:
                    self.reads.entry(g, head, stream=stream)
                self.result = self.prepared
            else:
                raise schema_error()
        return None
