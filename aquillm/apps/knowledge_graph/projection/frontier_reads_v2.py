"""Meter every equality lookup and validate authenticated source records."""

from copy import deepcopy
from dataclasses import fields

from apps.knowledge_graph.retrieval.topology.contracts import TopologyFailureReason
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import (
    FrontierCompletedBudgetFailureV2,
)

from .frontier_cypher_v2 import (
    INDEX_BATCH_LOOKUP, INDEX_LOOKUP, MANIFEST_LOOKUP, RAW_LOOKUP,
)
from .frontier_index_contracts_v2 import decode_record, marker_present, record_digest
from .memgraph_driver import MemgraphDriverError
from .memgraph_pagination import FAMILY_IDENTITY_FIELDS
from .memgraph_records import FAMILIES, _dto

QUERIES = {
    "manifest": MANIFEST_LOOKUP, "index": INDEX_LOOKUP,
    "index_batch": INDEX_BATCH_LOOKUP, "raw": RAW_LOOKUP,
}
KINDS = dict(FAMILIES)


def schema_error(message="frontier source schema mismatch"):
    return TopologyLoadError(TopologyFailureReason.BACKEND_SCHEMA_MISMATCH)


class FrontierReadsV2:
    def __init__(self, driver, *, ready, budget, deadline):
        self.driver, self.ready, self.budget = driver, ready, budget
        self.deadline = min(deadline, budget.deadline)
        self.manifests = {}
        self._index_records = {}

    def read(self, kind, generation, keys=()):
        maximum = 10 * len(keys) if kind == "raw" else 2
        if kind == "index_batch":
            if not 1 <= len(keys) <= 32:
                raise ValueError("frontier index batch exceeds its bound")
            maximum = 2 * len(keys)
        if maximum > 320 or maximum < 1:
            raise ValueError("frontier lookup batch exceeds its bound")
        if (
            kind != "manifest"
            and self.budget.queries + len(self.ready.selected_generations)
            >= self.budget.max_queries
        ):
            raise FrontierBudgetExceeded(
                "frontier reserves final authority verification"
            )
        self.budget.reserve_read(maximum)
        remaining = min(
            self.deadline - self.budget.clock(),
            self.budget.turn_budget.remaining_ms() / 1000,
        )
        if remaining <= 0:
            raise TimeoutError("frontier original deadline expired")
        parameters = {"generation_key": generation}
        if kind == "index":
            parameters = {"index_key": keys[0]}
        elif kind in {"raw", "index_batch"}:
            parameters["keys_csv"] = ",".join(keys)
        try:
            if callable(getattr(self.driver, "execute_frontier_read", None)):
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
        rows = self.read("manifest", generation)
        if len(rows) != 1 or type(rows[0]) is not dict or set(rows[0]) != {"record"}:
            raise schema_error()
        marker = rows[0]["record"]
        if type(marker) is not dict:
            raise schema_error()
        names = (
            "collection_key",
            "generation_key",
            "projection_key",
            "graph_checksum",
            "schema_version",
            "projection_version",
            "identifier_key_version",
            "membership_epoch",
            "membership_checksum",
        )
        if any(marker.get(n) != getattr(selected, n) for n in names) or (
            marker.get("state") != "ready"
            or marker.get("artifact_key") != selected.active_artifact_key
        ):
            raise TopologyLoadError(TopologyFailureReason.READINESS_MISMATCH)
        try:
            indexed = marker_present(marker)
        except (ValueError, TypeError):
            raise schema_error() from None
        if final:
            if self.manifests.get(generation) != marker:
                raise TopologyLoadError(TopologyFailureReason.READINESS_MISMATCH)
        else:
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
            if len(rows) != 1 or type(rows[0]) is not dict or set(rows[0]) != {"record"}:
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
        key = self.manifests[generation]["frontier_root_key"]
        seen, low, high = set(), "", "\uffff"
        while key is not None:
            if key in seen or len(seen) > 32:
                raise schema_error()
            seen.add(key)
            kind, node = self.index(generation, key)
            if (
                kind != "directory"
                or type(node) is not dict
                or set(node) != {"name", "head", "left", "right"}
            ):
                raise schema_error()
            current = node["name"]
            if type(current) is not str or not low < current < high:
                raise schema_error()
            if current == name:
                return node["head"]
            if name < current:
                key, high = node["left"], current
            else:
                key, low = node["right"], current
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
        actual, occupancies = {}, {}
        for row in rows:
            if type(row) is not dict or set(row) != {"lookup_key", "record", "labels"}:
                raise schema_error()
            key, properties, labels = row["lookup_key"], row["record"], row["labels"]
            if (
                key not in keys
                or type(properties) is not dict
                or type(labels) is not list
            ):
                raise schema_error()
            occupancies[key] = occupancies.get(key, 0) + 1
            families = set(labels) & set(KINDS)
            if (
                occupancies[key] > 9
                or len(families) != 1
                or set(labels) != families | {"ProjectedRecord"}
            ):
                raise schema_error()
            label = next(iter(families))
            if (
                (label, key) in actual
                or properties.get("generation_key") != generation
                or properties.get("opaque_key") != key
            ):
                raise schema_error()
            try:
                names = {f.name for f in fields(KINDS[label])} | {
                    "opaque_key",
                    "generation_key",
                }
                if set(properties) - names:
                    raise ValueError
                value = _dto(KINDS[label], properties)
                if getattr(value, FAMILY_IDENTITY_FIELDS[label]) != key:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise schema_error() from None
            actual[label, key] = value
        if any(
            k not in actual or record_digest(actual[k]) != v
            for k, v in expected.items()
        ):
            raise schema_error()
        return {k: actual[k] for k in expected}
