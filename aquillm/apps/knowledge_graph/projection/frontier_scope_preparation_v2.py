"""Finite breadth-wave scope preparation; whole-scope publication stays outside."""

from .frontier_directory_proof_v2 import directory_step
from .frontier_preparation_reads_v2 import PreparationReads
from .frontier_reads_v2 import schema_error


class ScopePreparation(PreparationReads):
    def __init__(self, reads, sources):
        super().__init__(reads)
        self.stage = "scope_directory"
        self.pending = {
            (g, "scope/" + source): (
                reads.manifests[g]["frontier_root_key"],
                set(),
                "",
                "\uffff",
                0,
            )
            for g, keys in sources.items()
            for source in keys
        }
        self.heads = {}

    def next_read(self, allowance=None):
        while self.result is None:
            self.reads._check_index_deadline()
            if self.stage == "scope_directory":
                if not self.pending:
                    self.stage = "scope_entries"
                    continue
                if self.wave is None:
                    for key, seen, _, _, prior in self.pending.values():
                        if key in seen or prior + len(seen) > 32:
                            raise schema_error()
                        seen.add(key)
                effect = self.index_wave(
                    ((g, s[0]) for (g, _), s in self.pending.items()), allowance
                )
                if effect:
                    return effect
                following = {}
                for (g, name), (key, seen, low, high, prior) in self.pending.items():
                    found, child, low, high = directory_step(
                        self.reads._index_records[g, key], name, low, high
                    )
                    if found:
                        if child is None:
                            raise schema_error()
                        self.heads[g, name] = child
                        continue
                    if child is None:
                        raise schema_error()
                    following[g, name] = child, seen, low, high, prior
                self.pending = following
            elif self.stage == "scope_entries":
                effect = self.index_wave(
                    ((g, head) for (g, _), head in self.heads.items()), allowance
                )
                if effect:
                    return effect
                for (g, stream), head in self.heads.items():
                    entry = self.reads.entry(g, head, stream=stream)
                    if entry["family"] != "scope" or entry["next"] is not None:
                        raise schema_error()
                    self.add_refs(g, entry["refs"])
                self.stage = "scope_raw"
            elif self.stage == "scope_raw":
                effect = self.raw_read(allowance)
                if effect:
                    return effect
                self.result = self.fresh
            else:
                raise schema_error()
        return None
