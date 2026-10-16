"""Scope-only digest container; Task2 binary leaf bytes are unchanged."""

from .frontier_binary_codec_v2 import decode_binary_response
from .frontier_contracts_v2 import _key

MAGIC = b"KGFS0001"
KINDS = {"index_batch_multi": 1, "scope_raw_complete_batch": 3}
MAXIMUM = 1048576


def decode_container(body, request_digest, *, single=False):
    _key(request_digest)
    if (
        type(body) is not bytes
        or not 41 <= len(body) <= MAXIMUM
        or body[:8] != MAGIC
        or body[8:40] != bytes.fromhex(request_digest)
        or not 1 <= body[40] <= 8
        or (single and body[40] != 1)
    ):
        raise ValueError("frontier scope window header invalid")
    result, offset, raw, rows, expanded = [], 41, 0, 0, 0
    for _ in range(body[40]):
        if len(body) - offset < 5 or body[offset] not in (1, 3):
            raise ValueError("frontier scope frame header invalid")
        is_raw = body[offset] == 3
        if single and not is_raw:
            raise ValueError("frontier single requires complete raw frame")
        kind = "scope_raw_complete_batch" if is_raw else "index_batch_multi"
        size = int.from_bytes(body[offset + 1 : offset + 5], "big")
        offset += 5
        if size < 12 or size > len(body) - offset:
            raise ValueError("frontier scope frame length invalid")
        frame = body[offset : offset + size]
        expanded += int.from_bytes(frame[8:12], "big")
        raw += is_raw
        if expanded > MAXIMUM or raw > 4:
            raise ValueError("frontier scope aggregate bound exceeded")
        payload = decode_binary_response(frame)
        count = payload["scanned_rows"]
        rows += count
        if rows > 320 or (is_raw and count >= 320):
            raise ValueError("frontier scope complete row bound exceeded")
        result.append((kind, tuple(payload["rows"])))
        offset += size
    if offset != len(body):
        raise ValueError("frontier scope trailing bytes")
    return tuple(result)


def encode_container(request_digest, frames, *, single=False):
    _key(request_digest)
    if not 1 <= len(frames) <= 8:
        raise ValueError("frontier scope frame count invalid")
    result = MAGIC + bytes.fromhex(request_digest) + bytes([len(frames)])
    for kind, frame in frames:
        if kind not in KINDS or type(frame) is not bytes:
            raise ValueError("frontier scope frame invalid")
        result += bytes([KINDS[kind]]) + len(frame).to_bytes(4, "big") + frame
    decode_container(result, request_digest, single=single)
    return result
