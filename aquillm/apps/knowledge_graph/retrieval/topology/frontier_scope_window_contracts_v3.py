"""Two exact scope-only requests over the original finite-window grant."""

from dataclasses import asdict, dataclass
from hashlib import sha256

from .frontier_contracts_v2 import _key
from .frontier_window_contracts_v2 import (
    decode_grant,
    decode_ready,
    encode_value,
)
from .frontier_window_validation_v2 import array, exact, validate_checkpoint
from .gateway_contracts import MAX_REQUEST_BYTES
from .gateway_json import load_gateway_json

PATH = "/v3/topology/frontier/scope-window"
VERSION = "topology-frontier-scope-window-v3"
CHECKSUM = sha256(
    b"scope-complete-window-v3:mode-window-or-single:ready-operation-checkpoint-grant-sequence:scope-only:KGFS0001:sha256:index160-limit2:raw128-limit10-total320-reject320:raw-reserve320:frames8:raw4:rows320:expanded1048576:wire1048576:request4194304:owner-replay:single-predispatch-checkpoint-only:captured-directory-recipes:stage-byte-bounds"
).hexdigest()


def protocol_headers_valid(names, *, request=False):
    allowed = {"x-topology-schema-version", "x-topology-schema-checksum"}
    if request:
        allowed.add("x-frontier-timeout-ms")
    return all(
        not name.lower().startswith(("x-topology-", "x-frontier-"))
        or name.lower() in allowed
        for name in names
    )


@dataclass(frozen=True)
class ScopeWindowRequest:
    ready: object
    mode: str
    operation: dict
    checkpoint: dict | None
    grant: object
    sequence: int
    request_digest: str


@dataclass(frozen=True)
class ScopeRawRequest:
    lookups: tuple
    ready: object
    response_bytes: int = 1048576
    kind: str = "scope_raw_complete_batch"
    maximum: int = 320


def validate_scope_checkpoint(ready, operation, checkpoint):
    exact(operation, {"kind", "roots", "request_digest"})
    if operation["kind"] != "scope_complete":
        raise ValueError("frontier complete scope operation invalid")
    validate_checkpoint(ready, {**operation, "kind": "scope"}, checkpoint)


def single_pairs(ready, operation):
    exact(operation, {"kind", "request_digest", "lookups"})
    if operation["kind"] != "scope_raw_complete":
        raise ValueError("frontier complete raw operation invalid")
    _key(operation["request_digest"])
    pairs = array(operation["lookups"], 128, 2)
    generations = [g.generation_key for g in ready.selected_generations]
    rebuilt = []
    for slot, key in pairs:
        if type(slot) is not int or not 0 <= slot < len(generations):
            raise ValueError("frontier generation slot invalid")
        _key(key)
        rebuilt.append((generations[slot], key))
    if not rebuilt or rebuilt != sorted(set(rebuilt)):
        raise ValueError("frontier complete raw pairs invalid")
    return tuple(rebuilt)


def encode_request(ready, operation, checkpoint, grant, sequence, check, *, pairs=None):
    mode = "window" if pairs is None else "single"
    if pairs is not None:
        slots = {g.generation_key: i for i, g in enumerate(ready.selected_generations)}
        operation = {
            "kind": "scope_raw_complete",
            "request_digest": operation["request_digest"],
            "lookups": [[slots[g], key] for g, key in pairs],
        }
        single_pairs(ready, operation)
    else:
        validate_scope_checkpoint(ready, operation, checkpoint)
    decode_grant(asdict(grant), sequence)
    value = dict(
        ready=asdict(ready),
        mode=mode,
        operation=operation,
        grant=asdict(grant),
        sequence=sequence,
    )
    if mode == "window":
        value["checkpoint"] = checkpoint
    return encode_value(value, check)


def decode_request(body, check=lambda: None):
    check()
    value = load_gateway_json(
        body, MAX_REQUEST_BYTES, lambda value: encode_value(value, check), check=check
    )
    check()
    if type(value) is not dict or value.get("mode") not in {"window", "single"}:
        raise ValueError("frontier scope mode invalid")
    window = value["mode"] == "window"
    exact(
        value,
        {"ready", "mode", "operation", "grant", "sequence"}
        | ({"checkpoint"} if window else set()),
    )
    ready = decode_ready(value["ready"])
    grant = decode_grant(value["grant"], value["sequence"])
    if window:
        validate_scope_checkpoint(ready, value["operation"], value["checkpoint"])
    else:
        single_pairs(ready, value["operation"])
    check()
    return ScopeWindowRequest(
        ready,
        value["mode"],
        value["operation"],
        value.get("checkpoint"),
        grant,
        value["sequence"],
        sha256(body).hexdigest(),
    )
