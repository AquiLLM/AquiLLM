"""Meter every equality lookup and validate authenticated source records."""

from copy import deepcopy

from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
    FrontierCompletedBudgetFailureV2,
    FrontierGatewayBatchRequestV2,
)

from .frontier_cypher_v2 import (
    INDEX_BATCH_LOOKUP,
    INDEX_BATCH_MULTI_LOOKUP,
    INDEX_LOOKUP,
    MANIFEST_BATCH_LOOKUP,
    MANIFEST_LOOKUP,
    RAW_BATCH_MULTI_LOOKUP,
    RAW_LOOKUP,
)
from .frontier_index_contracts_v2 import decode_record
from .frontier_source_validation_v2 import KINDS as KINDS
from .frontier_source_validation_v2 import schema_error as schema_error
from .frontier_source_validation_v2 import validate_marker, validate_raw
from .memgraph_driver import MemgraphDriverError

QUERIES = {
    "manifest": MANIFEST_LOOKUP,
    "index": INDEX_LOOKUP,
    "index_batch": INDEX_BATCH_LOOKUP,
    "raw": RAW_LOOKUP,
    "manifest_batch": MANIFEST_BATCH_LOOKUP,
    "index_batch_multi": INDEX_BATCH_MULTI_LOOKUP,
    "raw_batch_multi": RAW_BATCH_MULTI_LOOKUP,
}


class FrontierReadsV2:
    def __init__(self, driver, *, ready, budget, deadline):
        self.driver, self.ready, self.budget = driver, ready, budget
        self.deadline = min(deadline, budget.deadline)
        self.manifests = {}
        self._index_records = {}
        self._final_authority = None

    def reserve_final(self, *, multi):
        if self._final_authority is None:
            count = len(self.ready.selected_generations)
            self._final_authority = self.budget.reserve_authority(
                2 * count, (count + 159) // 160 if multi else count
            )

    def reserve_final_wire(self, rows, *, multi):
        if self._final_authority is None:
            return
        from apps.knowledge_graph.retrieval.topology.frontier_gateway_codec_v2 import (
            encode_batch_response,
        )
        from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical

        ordered = sorted(rows, key=_canonical) if multi else rows
        payload = _canonical({"rows": ordered, "scanned_rows": len(rows), "queries": 1})
        wire = encode_batch_response(payload) if multi else payload
        self.budget.reserve_authority_wire(self._final_authority, len(wire))

    def read_batch(self, kind, lookups, *, final=False):
        request = FrontierGatewayBatchRequestV2(
            kind,
            lookups,
            self.ready,
            max(1, self.budget.max_wire_bytes - self.budget.wire_bytes),
        )
        return self.read(
            kind, None, lookups=lookups, maximum=request.maximum, final=final
        )

    def read(
        self, kind, generation, keys=(), *, lookups=None, maximum=None, final=False
    ):
        maximum = (
            maximum if lookups is not None else (10 * len(keys) if kind == "raw" else 2)
        )
        if kind == "index_batch":
            if not 1 <= len(keys) <= 32:
                raise ValueError("frontier index batch exceeds its bound")
            maximum = 2 * len(keys)
        if maximum > 320 or maximum < 1:
            raise ValueError("frontier lookup batch exceeds its bound")
        if not final and self.budget.available_wire_bytes() <= 0:
            raise FrontierBudgetExceeded("frontier reserves final wire authority")
        if (
            kind != "manifest"
            and lookups is None
            and self._final_authority is None
            and self.budget.queries + len(self.ready.selected_generations)
            >= self.budget.max_queries
        ):
            raise FrontierBudgetExceeded(
                "frontier reserves final authority verification"
            )
        self.budget.reserve_read(
            maximum, authority=self._final_authority if final else None
        )
        if final:
            self._final_authority = None
        remaining = min(
            self.deadline - self.budget.clock(),
            self.budget.turn_budget.remaining_ms() / 1000,
        )
        if remaining <= 0:
            raise TimeoutError("frontier original deadline expired")
        parameters = {"generation_key": generation}
        if lookups is not None:
            parameters = {"lookups_csv": ",".join(g + ":" + k for g, k in lookups)}
        elif kind == "index":
            parameters = {"index_key": keys[0]}
        elif kind in {"raw", "index_batch"}:
            parameters["keys_csv"] = ",".join(keys)
        try:
            if lookups is not None and callable(
                getattr(self.driver, "execute_frontier_batch", None)
            ):
                rows = self.driver.execute_frontier_batch(
                    kind=kind,
                    lookups=lookups,
                    ready=self.ready,
                    deadline=self.deadline,
                    budget=self.budget,
                )
            elif lookups is None and callable(
                getattr(self.driver, "execute_frontier_read", None)
            ):
                rows = self.driver.execute_frontier_read(
                    kind=kind,
                    generation=generation,
                    keys=keys,
                    ready=self.ready,
                    deadline=self.deadline,
                    budget=self.budget,
                )
            else:
                rows = self.driver.execute_read(
                    QUERIES[kind],
                    parameters,
                    timeout_seconds=float(remaining),
                    max_records=maximum,
                )
        except FrontierCompletedBudgetFailureV2 as error:
            self.budget.charge_read(
                reserved_rows=maximum, scanned_rows=error.scanned_rows
            )
            if error.scanned_rows > maximum:
                raise schema_error() from error
            raise
        except MemgraphDriverError as error:
            self.budget.stop_uncertain()
            from .topology_adapter import _failure

            if error.code == "memgraph_timeout":
                raise TimeoutError("frontier datastore timeout") from error
            raise TopologyLoadError(_failure(error.code)) from error
        except BaseException:
            # A failed transport can leave real work running. Never continue the ledger.
            self.budget.stop_uncertain()
            raise
        scanned = len(rows) if type(rows) is tuple else maximum
        charged = self.budget.charge_read(reserved_rows=maximum, scanned_rows=scanned)
        if not charged or type(rows) is not tuple or len(rows) > maximum:
            raise schema_error()
        self.budget.check_active()
        if self.budget.clock() >= self.deadline:
            raise TimeoutError("frontier result arrived after its deadline")
        return rows

    def attest(self, selected, *, final=False):
        generation = selected.generation_key
        rows = self.read("manifest", generation, final=final)
        if len(rows) != 1 or type(rows[0]) is not dict or set(rows[0]) != {"record"}:
            raise schema_error()
        marker = rows[0]["record"]
        indexed = self.validate_marker(selected, marker, final=final)
        if not final:
            self.reserve_final_wire(rows, multi=False)
        return indexed

    def validate_marker(self, selected, marker, *, final=False, store=True):
        generation = selected.generation_key
        indexed = validate_marker(
            selected, marker, initial=self.manifests.get(generation), final=final
        )
        if not final and store:
            self.manifests[generation] = marker
        return indexed

    def _check_index_deadline(self):
        self.budget.check_active()
        if self.budget.clock() >= self.deadline:
            raise TimeoutError("frontier original deadline expired")

    def index(self, generation, key):
        self._check_index_deadline()
        identity = generation, key
        if identity not in self._index_records:
            rows = self.read("index", generation, (key,))
            if (
                len(rows) != 1
                or type(rows[0]) is not dict
                or set(rows[0]) != {"record"}
            ):
                raise schema_error()
            try:
                decoded = decode_record(
                    rows[0]["record"], expected_key=key, generation=generation
                )
            except (KeyError, TypeError, ValueError):
                raise schema_error() from None
            # Immutable, content-addressed data only, bounded by admitted reads.
            # Every authority attestation and raw evidence read remains fresh.
            self._index_records[identity] = decoded
        result = deepcopy(self._index_records[identity])
        self._check_index_deadline()
        return result

    def indices(self, generation, keys):
        from .frontier_index_batch_v2 import read_indices

        return read_indices(self, generation, keys)

    def head(self, generation, name):
        from .frontier_directory_proof_v2 import directory_step

        key = self.manifests[generation]["frontier_root_key"]
        seen, low, high = set(), "", "\uffff"
        while key is not None:
            if key in seen or len(seen) > 32:
                raise schema_error()
            seen.add(key)
            found, key, low, high = directory_step(
                self.index(generation, key), name, low, high
            )
            if found:
                return key
        return None

    def entry(self, generation, key, *, stream, previous=None):
        kind, value = self.index(generation, key)
        if (
            kind != "entry"
            or type(value) is not dict
            or set(value) != {"stream", "order", "refs", "links", "next", "family"}
        ):
            raise schema_error()
        order = value["order"]
        if (
            value["stream"] != stream
            or type(order) is not list
            or not 1 <= len(order) <= 8
            or any(type(v) is not str or len(v) > 256 for v in order)
            or (previous is not None and tuple(order) <= previous)
        ):
            raise schema_error()
        if (
            type(value["refs"]) is not list
            or not 1 <= len(value["refs"]) <= 32
            or type(value["links"]) is not dict
        ):
            raise schema_error()
        return value

    def hydrate(self, generation, refs):
        expected = {}
        for ref in refs:
            if type(ref) is not list or len(ref) != 3 or ref[0] not in KINDS:
                raise schema_error()
            label, key, checksum = ref
            if (label, key) in expected and expected[label, key] != checksum:
                raise schema_error()
            expected[label, key] = checksum
        keys = tuple(sorted({k for _, k in expected}))
        rows = self.read("raw", generation, keys)
        return self.validate_raw(generation, expected, keys, rows)

    def validate_raw(self, generation, expected, keys, rows):
        return validate_raw(generation, expected, keys, rows)
