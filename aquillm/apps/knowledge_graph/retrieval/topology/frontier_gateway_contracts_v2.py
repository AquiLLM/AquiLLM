"""Private fixed equality reads with relative time and strict bounded JSON."""

from dataclasses import asdict, dataclass
from hashlib import sha256

from .contracts import (
    AuthorizedProjectedDocumentV1,
    ReadyGenerationBundleV1,
    SelectedCollectionGenerationV1,
)
from .frontier_contracts_v2 import FrontierBudgetExceeded, _key
from .gateway_contracts import MAX_REQUEST_BYTES, _canonical, _load

PATH = "/v2/topology/frontier"
VERSION = "topology-frontier-v2"
CHECKSUM = sha256(
    b"frontier-v2:fixed-manifest-index-indexbatch32-raw:relative-timeout:raw-body:limit2:raw-limit10"
).hexdigest()
BATCH_PATH = PATH + "/batch"
BATCH_VERSION = "topology-frontier-batch-v2"
BATCH_CHECKSUM = sha256(
    b"frontier-batch-v2:generation-key-pairs:manifest160:index160:raw32:limit2:limit10:scalar-csv:relative-timeout:zlib1-base64-envelope:expanded1048576:strict-single-stream:canonical-row-order"
).hexdigest()


@dataclass(frozen=True, slots=True)
class FrontierGatewayBatchRequestV2:
    kind: str
    lookups: tuple[tuple[str, str], ...]
    ready: ReadyGenerationBundleV1
    response_bytes: int

    def __post_init__(self):
        if self.kind not in {"manifest_batch", "index_batch_multi", "raw_batch_multi"}:
            raise ValueError("unknown frontier batch operation")
        if type(self.ready) is not ReadyGenerationBundleV1:
            raise TypeError("frontier ready bundle must be exact")
        cap = 32 if self.kind == "raw_batch_multi" else 160
        if type(self.lookups) is not tuple or not 1 <= len(self.lookups) <= cap:
            raise ValueError("frontier tuple batch exceeds bound")
        generations = {s.generation_key for s in self.ready.selected_generations}
        for pair in self.lookups:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("frontier lookup pair must be exact")
            generation, key = pair
            _key(generation)
            _key(key)
            if generation not in generations or (
                self.kind == "manifest_batch" and generation != key
            ):
                raise ValueError("frontier tuple outside captured scope")
        if self.lookups != tuple(sorted(set(self.lookups))):
            raise ValueError("frontier tuples must be canonical and unique")
        if (
            type(self.response_bytes) is not int
            or not 1 <= self.response_bytes <= 1048576
        ):
            raise ValueError("frontier response allowance invalid")

    @property
    def maximum(self):
        return (10 if self.kind == "raw_batch_multi" else 2) * len(self.lookups)


def decode_batch_request(body):
    value = _load(body, maximum=MAX_REQUEST_BYTES)
    if set(value) != {"kind", "lookups", "ready", "response_bytes"}:
        raise ValueError("frontier batch request shape differs")
    if type(value["lookups"]) is not list or any(
        type(p) is not list for p in value["lookups"]
    ):
        raise ValueError("frontier wire lookups must be arrays")
    # Share the strict existing ready decoder without widening the old route.
    ready = value["ready"]
    generation = ready["selected_generations"][0]["generation_key"]
    decoded = decode_request(
        _canonical(
            {
                "kind": "manifest",
                "generation": generation,
                "keys": [],
                "ready": ready,
                "response_bytes": value["response_bytes"],
            }
        )
    )
    return FrontierGatewayBatchRequestV2(
        value["kind"],
        tuple(tuple(p) for p in value["lookups"]),
        decoded.ready,
        value["response_bytes"],
    )


class FrontierCompletedBudgetFailureV2(FrontierBudgetExceeded):
    """A completed datastore read whose response cannot fit its wire allowance."""

    def __init__(self, scanned_rows):
        if type(scanned_rows) is not int or not 0 <= scanned_rows <= 320:
            raise ValueError("frontier completed row charge is invalid")
        self.scanned_rows = scanned_rows
        super().__init__("frontier completed response exceeds wire allowance")


@dataclass(frozen=True, slots=True)
class FrontierGatewayRequestV2:
    kind: str
    generation: str
    keys: tuple[str, ...]
    ready: ReadyGenerationBundleV1
    response_bytes: int

    def __post_init__(self):
        if self.kind not in {"manifest", "index", "index_batch", "raw"}:
            raise ValueError("unknown frontier operation")
        _key(self.generation)
        if type(self.ready) is not ReadyGenerationBundleV1:
            raise TypeError("frontier ready bundle must be exact")
        if self.generation not in {
            g.generation_key for g in self.ready.selected_generations
        }:
            raise ValueError("frontier generation outside captured scope")
        if type(self.keys) is not tuple or any(type(k) is not str for k in self.keys):
            raise TypeError("frontier keys must be exact")
        for key in self.keys:
            _key(key)
        if self.keys != tuple(sorted(set(self.keys))) or not (
            (self.kind == "manifest" and not self.keys)
            or (self.kind == "index" and len(self.keys) == 1)
            or (self.kind in {"raw", "index_batch"} and 1 <= len(self.keys) <= 32)
        ):
            raise ValueError("frontier lookup keys exceed exact shape")
        if (
            type(self.response_bytes) is not int
            or not 1 <= self.response_bytes <= 1048576
        ):
            raise ValueError("frontier response allowance invalid")

    @property
    def maximum(self):
        if self.kind == "index_batch":
            return 2 * len(self.keys)
        return 10 * len(self.keys) if self.kind == "raw" else 2


def encode_request(request):
    body = _canonical(asdict(request))
    if len(body) > MAX_REQUEST_BYTES:
        raise ValueError("frontier request envelope exceeds existing bound")
    return body


def decode_request(body):
    value = _load(body, maximum=MAX_REQUEST_BYTES)
    if set(value) != {"kind", "generation", "keys", "ready", "response_bytes"}:
        raise ValueError("frontier request shape differs")
    ready = value["ready"]
    if type(ready) is not dict or set(ready) != {
        "selected_generations",
        "authorized_documents",
        "authorization_context_signature",
        "bundle_checksum",
    }:
        raise ValueError("frontier ready shape differs")
    if type(value["keys"]) is not list:
        raise ValueError("frontier wire keys must be an array")
    return FrontierGatewayRequestV2(
        value["kind"],
        value["generation"],
        tuple(value["keys"]),
        ReadyGenerationBundleV1(
            tuple(
                SelectedCollectionGenerationV1(**r)
                for r in ready["selected_generations"]
            ),
            tuple(
                AuthorizedProjectedDocumentV1(**r)
                for r in ready["authorized_documents"]
            ),
            ready["authorization_context_signature"],
            ready["bundle_checksum"],
        ),
        value["response_bytes"],
    )
