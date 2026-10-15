"""A fixed finite-operation request and correlated completed-work receipt."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

from .contracts import (
    AuthorizedProjectedDocumentV1,
    ReadyGenerationBundleV1,
    SelectedCollectionGenerationV1,
)
from .frontier_contracts_v2 import FrontierBudgetExceeded, _key
from .frontier_window_budget_v2 import WindowGrant
from .frontier_window_validation_v2 import exact, validate_checkpoint
from .gateway_contracts import MAX_REQUEST_BYTES, _canonical, _load

PATH = "/v2/topology/frontier/window"
VERSION = "topology-frontier-window-v2"
CHECKSUM = sha256(
    b"finite-scope-copy-v2:ready-checkpoint-grant-sequence:pair-mask25600:KGFW0001:sha256:frames8:raw4:rows320:expanded1048576:wire1048576:request4194304:owner-replay"
).hexdigest()


class CheckpointTooLarge(ValueError):
    """Only a pre-dispatch transport choice may catch this error."""


@dataclass(frozen=True)
class WindowRequest:
    ready: ReadyGenerationBundleV1
    operation: dict
    checkpoint: dict
    grant: WindowGrant
    sequence: int
    request_digest: str


class CompletedWindowBudgetFailure(FrontierBudgetExceeded):
    def __init__(self, request_digest, queries, scanned_rows):
        _key(request_digest)
        if (
            type(queries) is not int
            or not 1 <= queries <= 8
            or type(scanned_rows) is not int
            or not 0 <= scanned_rows <= 320 * queries
        ):
            raise ValueError("frontier completed window receipt invalid")
        self.request_digest, self.queries, self.scanned_rows = (
            request_digest,
            queries,
            scanned_rows,
        )
        super().__init__("frontier completed window exceeds allowance")


def encode_request(ready, operation, checkpoint, grant, sequence, check):
    validate_checkpoint(ready, operation, checkpoint)
    value = dict(
        ready=asdict(ready),
        operation=operation,
        checkpoint=checkpoint,
        grant=asdict(grant),
        sequence=sequence,
    )
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    chunks, size = [], 0
    check()
    for index, part in enumerate(encoder.iterencode(value)):
        if index % 256 == 0:
            check()
        chunk = part.encode("utf-8")
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise CheckpointTooLarge("frontier checkpoint exceeds request bound")
        chunks.append(chunk)
    check()
    return b"".join(chunks)


def decode_request(body):
    value = _load(body, maximum=MAX_REQUEST_BYTES)
    exact(value, {"ready", "operation", "checkpoint", "grant", "sequence"})
    data = value["ready"]
    exact(
        data,
        {
            "selected_generations",
            "authorized_documents",
            "authorization_context_signature",
            "bundle_checksum",
        },
    )
    if (
        type(data["selected_generations"]) is not list
        or not 1 <= len(data["selected_generations"]) <= 128
        or type(data["authorized_documents"]) is not list
        or len(data["authorized_documents"]) > 10000
    ):
        raise ValueError("frontier ready counts invalid")
    ready = ReadyGenerationBundleV1(
        tuple(
            SelectedCollectionGenerationV1(**g) for g in data["selected_generations"]
        ),
        tuple(AuthorizedProjectedDocumentV1(**d) for d in data["authorized_documents"]),
        data["authorization_context_signature"],
        data["bundle_checksum"],
    )
    grant = value["grant"]
    exact(grant, {"queries", "rows", "wire", "planning_rows"})
    if any(
        type(grant[k]) is not int or not 1 <= grant[k] <= maximum
        for k, maximum in {
            "queries": 8,
            "rows": 2560,
            "wire": 1048576,
            "planning_rows": 5000,
        }.items()
    ) or grant["rows"] != min(grant["planning_rows"], grant["queries"] * 320):
        raise ValueError("frontier window grant invalid")
    if type(value["sequence"]) is not int or not 1 <= value["sequence"] <= 64:
        raise ValueError("frontier window sequence invalid")
    validate_checkpoint(ready, value["operation"], value["checkpoint"])
    if body != _canonical(value):
        raise ValueError("frontier window request not canonical")
    return WindowRequest(
        ready,
        value["operation"],
        value["checkpoint"],
        WindowGrant(**grant),
        value["sequence"],
        sha256(body).hexdigest(),
    )
