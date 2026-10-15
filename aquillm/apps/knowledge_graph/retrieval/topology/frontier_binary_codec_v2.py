"""Versioned bounded binary envelope around the unchanged canonical batch JSON."""

import zlib

from .frontier_gateway_codec_v2 import MAX_EXPANDED_BYTES, validate_expanded_payload

MAGIC = b"KGFZ0001"


def encode_binary_response(payload):
    if type(payload) is not bytes or not 1 <= len(payload) <= MAX_EXPANDED_BYTES:
        raise ValueError("frontier expanded response exceeds bound")
    return MAGIC + len(payload).to_bytes(4, "big") + zlib.compress(payload, 1)


def decode_binary_response(body):
    if type(body) is not bytes or len(body) < 12 or body[:8] != MAGIC:
        raise ValueError("frontier binary header differs")
    size = int.from_bytes(body[8:12], "big")
    if not 1 <= size <= MAX_EXPANDED_BYTES:
        raise ValueError("frontier expanded response exceeds bound")
    try:
        stream = zlib.decompressobj()
        expanded = stream.decompress(body[12:], MAX_EXPANDED_BYTES + 1)
        if (
            len(expanded) != size
            or not stream.eof
            or stream.unused_data
            or stream.unconsumed_tail
        ):
            raise ValueError("frontier binary stream exceeds exact bound")
    except zlib.error:
        raise ValueError("frontier binary stream invalid") from None
    return validate_expanded_payload(expanded)
