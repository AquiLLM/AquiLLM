"""Physical effects and transactional waves shared by two finite preparations."""

from dataclasses import dataclass

from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
    _key,
)

from .frontier_generation_batch_v2 import keyed_rows
from .frontier_index_contracts_v2 import decode_record
from .frontier_reads_v2 import KINDS, schema_error


class OwnerBoundary(Exception):
    """The next transition needs values retained only by the original owner."""


@dataclass(frozen=True)
class PreparationRead:
    kind: str
    pairs: tuple

    @property
    def maximum(self):
        return len(self.pairs) * (10 if self.kind == "raw_batch_multi" else 2)


class PreparationReads:
    def __init__(self, reads):
        self.reads = reads
        self.guard = set()
        self.wave = None
        self.staged = {}
        self.expected = {}
        self.raw_pending = None
        self.fresh = {}
        self.earlier_raw = False
        self.effect = None
        self.result = None

    def _effect(self, kind, pending, allowance):
        factor, limit = (10, 32) if kind == "raw_batch_multi" else (2, 160)
        available = (
            self.reads.budget.available_read_rows() if allowance is None else allowance
        )
        count = min(limit, available // factor)
        if count < 1:
            raise FrontierBudgetExceeded("frontier physical work exhausted")
        self.effect = PreparationRead(kind, tuple(pending[:count]))
        return self.effect

    def index_wave(self, pairs, allowance):
        if self.wave is None:
            targets = tuple(sorted(set(pairs)))
            selected = {s.generation_key for s in self.reads.ready.selected_generations}
            for generation, key in targets:
                _key(generation)
                _key(key)
                if generation not in selected:
                    raise schema_error()
            missing = [
                p
                for p in targets
                if p not in self.reads._index_records and p not in self.guard
            ]
            self.wave = targets, missing
        targets, pending = self.wave
        if pending:
            return self._effect("index_batch_multi", pending, allowance)
        if any(
            p not in self.staged and p not in self.reads._index_records for p in targets
        ):
            raise OwnerBoundary
        self.reads._check_index_deadline()
        self.reads._index_records.update(self.staged)
        self.staged = {}
        self.wave = None
        return None

    def add_refs(self, generation, refs):
        for ref in refs:
            if type(ref) is not list or len(ref) != 3 or ref[0] not in KINDS:
                raise schema_error()
            label, key, checksum = ref
            _key(key)
            _key(checksum)
            target = generation, label, key
            if target in self.expected and self.expected[target] != checksum:
                raise schema_error()
            self.expected[target] = checksum

    def raw_read(self, allowance):
        if self.raw_pending is None:
            self.raw_pending = sorted({(g, key) for g, _, key in self.expected})
            self.fresh = {g: {} for g, _ in self.raw_pending}
        if self.raw_pending:
            return self._effect("raw_batch_multi", self.raw_pending, allowance)
        return None

    def apply_result(self, rows):
        self.reads._check_index_deadline()
        effect, self.effect = self.effect, None
        if effect is None:
            raise ValueError("preparation has no pending physical read")
        raw = effect.kind == "raw_batch_multi"
        grouped = keyed_rows(rows, effect.pairs, raw=raw)
        if not raw:
            decoded = {}
            for (generation, key), values in grouped.items():
                try:
                    value = decode_record(
                        values[0]["record"], expected_key=key, generation=generation
                    )
                    if value[0] not in {"directory", "entry"}:
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    raise schema_error() from None
                decoded[generation, key] = value
            self.staged.update(decoded)
            del self.wave[1][: len(effect.pairs)]
            return
        for generation in sorted({g for g, _ in effect.pairs}):
            keys = {k for g, k in effect.pairs if g == generation}
            refs = {
                (label, key): checksum
                for (g, label, key), checksum in self.expected.items()
                if g == generation and key in keys
            }
            physical = tuple(
                {k: v for k, v in row.items() if k != "generation_key"}
                for pair, values in grouped.items()
                if pair[0] == generation
                for row in values
            )
            self.fresh.setdefault(generation, {}).update(
                self.reads.validate_raw(generation, refs, keys, physical)
            )
        del self.raw_pending[: len(effect.pairs)]
