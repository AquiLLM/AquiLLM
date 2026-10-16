"""New scope schemas cannot be confused with legacy scope or copy programs."""

import json
from hashlib import sha256

import pytest

from apps.knowledge_graph.retrieval.topology import (
    frontier_scope_window_codec_v3 as codec,
)
from apps.knowledge_graph.retrieval.topology import (
    frontier_scope_window_contracts_v3 as contracts,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_codec_v2 import (
    decode_container as legacy_container,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_contracts_v2 import (
    decode_request as legacy_decode,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical
from apps.knowledge_graph.tests.frontier_scope_complete_support import (
    leaf,
    request_body,
    scope_state,
)


@pytest.mark.parametrize("single", [False, True])
def test_scope_requests_are_whole_body_correlated_and_legacy_rejected(single):
    fixture, state = scope_state(stage="scope_raw" if single else "scope_directory")
    body = request_body(fixture, state, single=single)
    decoded = contracts.decode_request(body)
    assert decoded.mode == ("single" if single else "window")
    assert decoded.request_digest == sha256(body).hexdigest()
    assert decoded.operation["request_digest"] != decoded.request_digest
    with pytest.raises(ValueError):
        legacy_decode(body)
    if single:
        assert (
            contracts.single_pairs(decoded.ready, decoded.operation)
            == state.effect.pairs
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "mixed",
        "copy",
        "stage",
        "duplicate",
        "foreign",
        "order",
        "slot",
        "limit",
        "grant",
        "sequence",
        "ready",
    ],
)
@pytest.mark.parametrize("single", [False, True])
def test_scope_request_rejects_mixed_or_unbounded_authority(single, mutation):
    fixture, state = scope_state(
        stage="scope_raw" if single else "scope_directory", generations=2
    )
    value = json.loads(request_body(fixture, state, single=single))
    operation = value["operation"]
    if mutation == "unknown":
        value["limit"] = 320
    elif mutation == "mixed":
        value["checkpoint"] = {} if single else value.pop("checkpoint")
        if not single:
            value["mode"] = "single"
    elif mutation == "copy":
        operation["kind"] = "copy"
    elif mutation == "stage":
        operation["roots" if single else "lookups"] = []
    elif mutation in {"duplicate", "foreign", "order", "slot", "limit"}:
        field = "lookups" if single else "roots"
        if mutation == "duplicate":
            operation[field].append(operation[field][0])
        elif mutation == "foreign":
            operation[field][0][1] = "Z" * 64
        elif mutation == "order":
            if single:
                operation[field].reverse()
            else:
                value["checkpoint"]["stage"] = "copy_raw"
        elif mutation == "slot":
            operation[field][0][0] = True
        elif single:
            operation[field] = [[0, f"{i:064x}"] for i in range(129)]
        else:
            value["checkpoint"]["guard"] = [[0, "a" * 64]] * 5001
    elif mutation == "grant":
        value["grant"]["rows"] -= 1
    elif mutation == "sequence":
        value["sequence"] = True
    else:
        value["ready"]["bundle_checksum"] = "a" * 64
    with pytest.raises((ValueError, TypeError)):
        contracts.decode_request(_canonical(value))


def test_json_duplicate_keys_and_noncanonical_text_fail_before_worker():
    fixture, state = scope_state()
    body = request_body(fixture, state)
    for changed in (body + b" ", b'{"mode":"window",' + body[1:]):
        with pytest.raises(ValueError):
            contracts.decode_request(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        "magic",
        "digest",
        "legacy_kind",
        "unknown_kind",
        "trailing",
        "raw320",
        "single_index",
        "single_two",
        "count",
        "rows",
        "expanded",
    ],
)
def test_container_rejects_legacy_or_incomplete_raw_authority(mutation):
    frame = leaf()
    kind, count, single = 3, 1, False
    if mutation == "raw320":
        frame = leaf(320)
    if mutation == "single_index":
        kind, single = 1, True
    if mutation == "single_two":
        count, single = 2, True
    if mutation == "count":
        count = 9
    if mutation == "rows":
        frame, count = leaf(161), 2
    if mutation == "expanded":
        frame, count = leaf(padding=550000), 2
    if mutation in {"legacy_kind", "unknown_kind"}:
        kind = 2 if mutation == "legacy_kind" else 4
    body = codec.MAGIC + bytes.fromhex("a" * 64) + bytes([count])
    body += (bytes([kind]) + len(frame).to_bytes(4, "big") + frame) * count
    if mutation == "magic":
        body = b"KGFW0001" + body[8:]
    if mutation == "digest":
        body = body[:8] + b"z" + body[9:]
    if mutation == "trailing":
        body += b"x"
    with pytest.raises(ValueError):
        codec.decode_container(body, "a" * 64, single=single)


def test_single_has_exactly_one_unchanged_binary_leaf():
    frame = leaf(319)
    body = codec.encode_container(
        "a" * 64, [("scope_raw_complete_batch", frame)], single=True
    )
    assert len(codec.decode_container(body, "a" * 64, single=True)[0][1]) == 319
    with pytest.raises(ValueError):
        legacy_container(body, "a" * 64)
