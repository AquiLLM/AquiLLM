"""Lossless batch envelope; wire and expanded JSON have independent bounds."""

import zlib
from base64 import b64decode, b64encode
from binascii import Error as Base64Error

from .frontier_contracts_v2 import _key
from .gateway_contracts import _canonical, _load

MAX_EXPANDED_BYTES = 1048576


def encode_batch_response(payload):
    if type(payload) is not bytes or len(payload) > MAX_EXPANDED_BYTES:
        raise ValueError("frontier expanded response exceeds bound")
    return _canonical(
        {
            "encoding": "zlib-base64",
            "payload": b64encode(zlib.compress(payload, 1)).decode("ascii"),
        }
    )


def decode_batch_response(value):
    if (
        type(value) is not dict
        or set(value) != {"encoding", "payload"}
        or value["encoding"] != "zlib-base64"
        or type(value["payload"]) is not str
    ):
        raise ValueError("frontier compressed envelope differs")
    try:
        encoded = value["payload"].encode("ascii")
        compressed = b64decode(encoded, validate=True)
        if b64encode(compressed) != encoded:
            raise ValueError("frontier base64 is not canonical")
        stream = zlib.decompressobj()
        expanded = stream.decompress(compressed, MAX_EXPANDED_BYTES + 1)
        if (
            len(expanded) > MAX_EXPANDED_BYTES
            or not stream.eof
            or stream.unused_data
            or stream.unconsumed_tail
        ):
            raise ValueError("frontier compressed stream exceeds exact bound")
    except (Base64Error, UnicodeError, zlib.error):
        raise ValueError("frontier compressed stream invalid") from None
    return validate_expanded_payload(expanded)


def validate_expanded_payload(expanded):
    """One semantic contract for both lossless batch response envelopes."""
    result = _load(expanded, maximum=MAX_EXPANDED_BYTES)
    if (
        type(result) is not dict
        or set(result) != {"rows", "scanned_rows", "queries"}
        or type(result["rows"]) is not list
        or len(result["rows"]) > 320
        or type(result["scanned_rows"]) is not int
        or result["scanned_rows"] != len(result["rows"])
        or type(result["queries"]) is not int
        or result["queries"] != 1
    ):
        raise ValueError("frontier expanded physical accounting differs")
    for row in result["rows"]:
        if (
            type(row) is not dict
            or set(row)
            not in (
                {"generation_key", "lookup_key", "record"},
                {"generation_key", "lookup_key", "record", "labels"},
            )
            or type(row["record"]) is not dict
            or (
                "labels" in row
                and (
                    type(row["labels"]) is not list
                    or any(type(label) is not str for label in row["labels"])
                )
            )
        ):
            raise ValueError("frontier expanded record shape differs")
        _key(row["generation_key"])
        _key(row["lookup_key"])
    return result
