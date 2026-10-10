"""Canonical V2 channel RRF. V1 baseline-prefix contracts remain independent."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Literal

from lib.embeddings.document_contracts import digest

ChannelV2 = Literal[
    "dense", "exact", "trigram", "full_text", "graph_direct", "graph_extended"
]
CHANNELS = ("dense", "exact", "trigram", "full_text", "graph_direct", "graph_extended")
MAX_CANDIDATES = 626


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("invalid_fusion_integer")


@dataclass(frozen=True, slots=True)
class ChannelRankV2:
    channel: ChannelV2
    rank: int

    def __post_init__(self):
        if type(self.channel) is not str or self.channel not in CHANNELS:
            raise ValueError("invalid_fusion_channel")
        integer(self.rank, 1, 20 if self.channel.startswith("graph_") else 606)


@dataclass(frozen=True, slots=True)
class FusedCandidateV2:
    chunk_id: int
    channel_ranks: tuple[ChannelRankV2, ...]
    rrf_score: float
    protected_exact: bool

    def __post_init__(self):
        integer(self.chunk_id, 1, 2**63 - 1)
        ranks = self.channel_ranks
        if (
            type(ranks) is not tuple
            or not ranks
            or any(type(r) is not ChannelRankV2 for r in ranks)
        ):
            raise ValueError("invalid_fusion_ranks")
        names = tuple(r.channel for r in ranks)
        if names != tuple(c for c in CHANNELS if c in names):
            raise ValueError("noncanonical_fusion_ranks")
        if (
            type(self.rrf_score) is not float
            or not isfinite(self.rrf_score)
            or self.rrf_score != fsum(1 / (60 + r.rank) for r in ranks)
        ):
            raise ValueError("invalid_fusion_score")
        if type(self.protected_exact) is not bool or (
            self.protected_exact and "exact" not in names
        ):
            raise ValueError("invalid_exact_protection")


def sort_key(row):
    return (
        -row.rrf_score,
        -len(row.channel_ranks),
        min(r.rank for r in row.channel_ranks),
        row.chunk_id,
    )


def checksum(candidates, policy_key, scope_key):
    return digest(
        {
            "version": "fusion-result-v2",
            "policy_key": policy_key,
            "scope_key": scope_key,
            "candidates": [
                {
                    "chunk_id": c.chunk_id,
                    "channel_ranks": [
                        {"channel": r.channel, "rank": r.rank} for r in c.channel_ranks
                    ],
                    "rrf_score": c.rrf_score.hex(),
                    "protected_exact": c.protected_exact,
                }
                for c in candidates
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class FusionResultV2:
    candidates: tuple[FusedCandidateV2, ...]
    policy_key: Literal["rrf60-v2", "rrf60-protected-exact-v2"]
    scope_key: str
    order_checksum: str

    def __post_init__(self):
        rows = self.candidates
        if (
            type(rows) is not tuple
            or len(rows) > MAX_CANDIDATES
            or any(type(c) is not FusedCandidateV2 for c in rows)
            or len({c.chunk_id for c in rows}) != len(rows)
        ):
            raise ValueError("invalid_fusion_candidates")
        if (
            type(self.scope_key) is not str
            or re.fullmatch("[0-9a-f]{64}", self.scope_key) is None
        ):
            raise ValueError("invalid_fusion_scope")
        if type(self.policy_key) is not str or self.policy_key not in (
            "rrf60-v2",
            "rrf60-protected-exact-v2",
        ):
            raise ValueError("invalid_fusion_policy")
        protected = tuple(c for c in rows if c.protected_exact)
        rest = tuple(c for c in rows if not c.protected_exact)
        if (
            len(protected) > 1
            or rows != protected + tuple(sorted(rest, key=sort_key))
            or (protected and self.policy_key != "rrf60-protected-exact-v2")
        ):
            raise ValueError("invalid_fusion_order")
        for channel in CHANNELS:
            ranks = [
                r.rank for c in rows for r in c.channel_ranks if r.channel == channel
            ]
            if len(set(ranks)) != len(ranks):
                raise ValueError("duplicate_fusion_rank")
        if type(self.order_checksum) is not str or self.order_checksum != checksum(
            rows, self.policy_key, self.scope_key
        ):
            raise ValueError("invalid_fusion_checksum")


def fuse_channel_ranks_v2(
    channels: Mapping[str, tuple[int, ...]],
    *,
    limit: int,
    scope_key: str,
    protected_exact_ids: tuple[int, ...] = (),
) -> FusionResultV2:
    integer(limit, 0, MAX_CANDIDATES)
    if not isinstance(channels, Mapping) or any(
        type(c) is not str or c not in CHANNELS for c in channels
    ):
        raise ValueError("invalid_fusion_channels")
    if type(protected_exact_ids) is not tuple or len(protected_exact_ids) > 1:
        raise ValueError("invalid_exact_protection")
    for pk in protected_exact_ids:
        integer(pk, 1, 2**63 - 1)
    memberships = {}
    for channel in CHANNELS:
        ids = channels.get(channel, ())
        if type(ids) is not tuple or len(ids) > (
            20 if channel.startswith("graph_") else 606
        ):
            raise ValueError("invalid_channel_candidates")
        unique = {}
        for pk in ids:
            integer(pk, 1, 2**63 - 1)
            unique.setdefault(pk, len(unique) + 1)
        for pk, rank in unique.items():
            memberships.setdefault(pk, []).append(ChannelRankV2(channel, rank))
    if any(pk not in channels.get("exact", ()) for pk in protected_exact_ids):
        raise ValueError("invalid_exact_protection")
    rows = [
        FusedCandidateV2(
            pk,
            tuple(ranks),
            fsum(1 / (60 + r.rank) for r in ranks),
            pk in protected_exact_ids,
        )
        for pk, ranks in memberships.items()
    ]
    rows.sort(key=lambda row: (not row.protected_exact, *sort_key(row)))
    selected = tuple(rows[:limit])
    policy = "rrf60-protected-exact-v2" if protected_exact_ids else "rrf60-v2"
    return FusionResultV2(
        selected, policy, scope_key, checksum(selected, policy, scope_key)
    )


def fuse_channel_ids(channels, *, limit, protected_exact_ids=()):
    result = fuse_channel_ranks_v2(
        channels,
        limit=limit,
        scope_key="0" * 64,
        protected_exact_ids=protected_exact_ids,
    )
    return tuple(c.chunk_id for c in result.candidates)
