"""Exact digest-bound container of independent Task2 binary frames."""

from .frontier_binary_codec_v2 import decode_binary_response
from .frontier_contracts_v2 import _key

MAGIC = b"KGFW0001"
KINDS = {"index_batch_multi": 1, "raw_batch_multi": 2}
MAXIMUM = 1048576


def encode_container(request_digest, frames):
    _key(request_digest)
    if not 1 <= len(frames) <= 8:
        raise ValueError("frontier window frame count invalid")
    result = MAGIC + bytes.fromhex(request_digest) + bytes([len(frames)])
    for kind, frame in frames:
        if kind not in KINDS or type(frame) is not bytes:
            raise ValueError("frontier window frame invalid")
        result += bytes([KINDS[kind]]) + len(frame).to_bytes(4, "big") + frame
    decode_container(result, request_digest)
    return result


def decode_container(body, request_digest):
    _key(request_digest)
    if (
        type(body) is not bytes
        or not 41 <= len(body) <= MAXIMUM
        or body[:8] != MAGIC
        or body[8:40] != bytes.fromhex(request_digest)
        or not 1 <= body[40] <= 8
    ):
        raise ValueError("frontier window header invalid")
    result, offset, raw, rows, expanded = [], 41, 0, 0, 0
    for _ in range(body[40]):
        if len(body) - offset < 5 or body[offset] not in (1, 2):
            raise ValueError("frontier window frame header invalid")
        kind = "index_batch_multi" if body[offset] == 1 else "raw_batch_multi"
        size = int.from_bytes(body[offset + 1 : offset + 5], "big")
        offset += 5
        if size < 12 or size > len(body) - offset:
            raise ValueError("frontier window frame length invalid")
        frame = body[offset : offset + size]
        expanded += int.from_bytes(frame[8:12], "big")
        raw += kind == "raw_batch_multi"
        if expanded > MAXIMUM or raw > 4:
            raise ValueError("frontier window aggregate bound exceeded")
        payload = decode_binary_response(frame)
        rows += payload["scanned_rows"]
        if rows > 320:
            raise ValueError("frontier window row bound exceeded")
        result.append((kind, tuple(payload["rows"])))
        offset += size
    if offset != len(body):
        raise ValueError("frontier window trailing bytes")
    return tuple(result)
