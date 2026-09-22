"""Opaque projected PPR ranking result and deterministic trace encoding."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProjectedPPRResultV1:
    """Opaque score vector, rank order, and provider-neutral trace bytes."""

    scores: tuple[tuple[str, float], ...]
    ranked_identity_keys: tuple[str, ...]
    trace_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.scores) is not tuple or any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not float
            for row in self.scores
        ):
            raise TypeError("scores must be exact opaque-key float pairs")
        keys = tuple(key for key, _ in self.scores)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("scores must be unique and opaque-key sorted")
        if (
            type(self.ranked_identity_keys) is not tuple
            or set(self.ranked_identity_keys) != set(keys)
            or len(self.ranked_identity_keys) != len(keys)
        ):
            raise ValueError("ranked identities must cover scores exactly")
        if type(self.trace_bytes) is not bytes:
            raise TypeError("trace_bytes must be exact bytes")


def _trace_bytes(
    scores: tuple[tuple[str, float], ...], ranked: tuple[str, ...]
) -> bytes:
    return json.dumps(
        {
            "ranked_identity_keys": ranked,
            "scores": [[key, score.hex()] for key, score in scores],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
