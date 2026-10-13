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
