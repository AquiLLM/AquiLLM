"""Deadline-checked byte capture primitives shared by native source storage."""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time

from .source_artifacts import SourceArtifactIntegrityError


CHUNK_BYTES = 65536


@dataclass(frozen=True)
class CapturedSource:
    data: bytes
    sha256: str
    size_bytes: int
    object_identity_sha256: str


def check_deadline(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise SourceArtifactIntegrityError("source_storage_deadline")


def _identity_digest(identity):
    return sha256(json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _capture(data, identity):
    return CapturedSource(data, sha256(data).hexdigest(), len(data), _identity_digest(identity))


def _read_bytes(read, length, deadline):
    captured = bytearray()
    while len(captured) < length:
        check_deadline(deadline)
        allowance = min(CHUNK_BYTES, length - len(captured))
        chunk = read(allowance)
        check_deadline(deadline)
        if not isinstance(chunk, bytes) or not chunk or len(chunk) > allowance:
            raise SourceArtifactIntegrityError("source_storage_length_mismatch")
        captured.extend(chunk)
    # A one-byte EOF probe detects a body that exceeds its declared length.
    check_deadline(deadline)
    extra = read(1)
    check_deadline(deadline)
    if extra != b"":
        raise SourceArtifactIntegrityError("source_storage_length_mismatch")
    return bytes(captured)
