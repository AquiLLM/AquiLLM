"""Closed numeric summaries of already prepared graph seed support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from .direct_seed_contracts import DirectResolutionTier, DirectSeedOutcomeV1
from .topology.contracts import ProjectedSeedV1
from .types import GraphExpansionSeed

_DIRECT_KEYS = frozenset(
    (
        "deduplicated_spans",
        "resolved_spans",
        "ambiguous_spans",
        "retained_matches",
        "retained_seeds",
        "exact_tier_matches",
        "minimum_extraction_score",
        "cap_pressure",
    )
)
_EXTENDED_KEYS = frozenset(
    (
        "requested_chunks",
        "mapped_top_chunks",
        "required_top_chunks",
        "rank_one_mapped",
        "rank_one_channel_agreement",
        "retained_seeds",
        "cap_pressure",
    )
)
_STATUSES = frozenset(("supported", "insufficient", "unknown"))


def _digest(
    status: str, cap_pressure: bool, summary: tuple[tuple[str, int | float | bool], ...]
) -> str:
    payload = {
        "domain": "ppr_seed_support_v1",
        "status": status,
        "cap_pressure": cap_pressure,
        "summary": [
            [key, value.hex() if type(value) is float else value]
            for key, value in summary
        ],
    }
    return sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PPRSeedSupportV1:
    status: str
    cap_pressure: bool
    summary: tuple[tuple[str, int | float | bool], ...]
    digest: str

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in _STATUSES:
            raise ValueError("unsupported seed support status")
        if type(self.cap_pressure) is not bool:
            raise TypeError("cap_pressure must be exact boolean")
        if type(self.summary) is not tuple or len(self.summary) > 16:
            raise TypeError("summary must be an exact bounded tuple")
        keys = []
        for row in self.summary:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("summary rows must be exact pairs")
            key, value = row
            if type(key) is not str or key not in _DIRECT_KEYS | _EXTENDED_KEYS:
                raise ValueError("summary key is not allowed")
            if type(value) not in (int, float, bool) or (
                type(value) is float and not isfinite(value)
            ):
                raise ValueError("summary values must be finite numeric or boolean")
            keys.append(key)
        if keys != sorted(set(keys)):
            raise ValueError("summary keys must be sorted and unique")
        if not (set(keys) <= _DIRECT_KEYS or set(keys) <= _EXTENDED_KEYS):
            raise ValueError("summary keys must belong to one branch")
        if self.digest != _digest(self.status, self.cap_pressure, self.summary):
            raise ValueError("summary digest does not bind its values")


@dataclass(frozen=True, slots=True)
class PreparedPPRSeedsV1:
    seeds: tuple[ProjectedSeedV1, ...]
    support: PPRSeedSupportV1
    intent: str

    def __post_init__(self) -> None:
        if type(self.seeds) is not tuple or any(
            type(seed) is not ProjectedSeedV1 for seed in self.seeds
        ):
            raise TypeError("seeds must be exact projected seed rows")
        if type(self.support) is not PPRSeedSupportV1:
            raise TypeError("support must be an exact PPRSeedSupportV1")
        if type(self.intent) is not str or self.intent not in (
            "focused",
            "balanced",
            "relational",
        ):
            raise ValueError("intent is unsupported")


def _support(
    status: str, cap_pressure: bool, values: dict[str, int | float | bool]
) -> PPRSeedSupportV1:
    summary = tuple(sorted(values.items()))
    return PPRSeedSupportV1(
        status, cap_pressure, summary, _digest(status, cap_pressure, summary)
    )


def summarize_direct_support(
    outcome: DirectSeedOutcomeV1, *, max_seeds: int
) -> PPRSeedSupportV1:
    if (
        type(outcome) is not DirectSeedOutcomeV1
        or type(max_seeds) is not int
        or not 1 <= max_seeds <= 64
    ):
        raise ValueError("direct support inputs are invalid")
    diagnostics = outcome.diagnostics
    cap_pressure = len(outcome.matches) >= max_seeds or len(outcome.seeds) >= max_seeds
    exact = sum(
        match.tier is not DirectResolutionTier.EMBEDDING for match in outcome.matches
    )
    values: dict[str, int | float | bool] = {
        "deduplicated_spans": diagnostics.deduplicated_span_count,
        "resolved_spans": diagnostics.resolved_span_count,
        "ambiguous_spans": diagnostics.ambiguous_span_count,
        "retained_matches": len(outcome.matches),
        "retained_seeds": len(outcome.seeds),
        "exact_tier_matches": exact,
        "cap_pressure": cap_pressure,
    }
    if not outcome.matches or outcome.failure_reason is not None:
        return _support("unknown", cap_pressure, values)
    minimum = min(match.extraction_confidence for match in outcome.matches)
    values["minimum_extraction_score"] = minimum
    supported = (
        not cap_pressure
        and diagnostics.ambiguous_span_count == 0
        and diagnostics.resolved_span_count == diagnostics.deduplicated_span_count
        and exact == len(outcome.matches)
        and minimum >= 0.80
    )
    return _support("supported" if supported else "insufficient", cap_pressure, values)


def summarize_extended_support(
    *,
    ranked_seeds: tuple[GraphExpansionSeed, ...],
    mapped_chunk_ids: frozenset[int],
    vector_chunk_ids: tuple[int, ...] | None,
    trigram_chunk_ids: tuple[int, ...] | None,
    exact_chunk_ids: tuple[int, ...] | None,
    retained_identity_count: int,
    max_seeds: int,
) -> PPRSeedSupportV1:
    if type(ranked_seeds) is not tuple or any(
        type(seed) is not GraphExpansionSeed for seed in ranked_seeds
    ):
        raise TypeError("ranked_seeds must be exact GraphExpansionSeed rows")
    if (
        type(mapped_chunk_ids) is not frozenset
        or type(retained_identity_count) is not int
        or type(max_seeds) is not int
        or not 1 <= max_seeds <= 64
        or retained_identity_count < 0
    ):
        raise ValueError("extended support inputs are invalid")
    cap_pressure = (
        len(ranked_seeds) >= max_seeds or retained_identity_count >= max_seeds
    )
    top = tuple(sorted(ranked_seeds, key=lambda seed: (seed.rank, seed.chunk_id)))[:3]
    mapped = sum(seed.chunk_id in mapped_chunk_ids for seed in top)
    values: dict[str, int | float | bool] = {
        "requested_chunks": len(ranked_seeds),
        "required_top_chunks": len(top),
        "mapped_top_chunks": mapped,
        "retained_seeds": retained_identity_count,
        "rank_one_mapped": bool(top and top[0].chunk_id in mapped_chunk_ids),
        "cap_pressure": cap_pressure,
    }
    channels = (vector_chunk_ids, trigram_chunk_ids, exact_chunk_ids)
    if not top or any(type(channel) is not tuple for channel in channels):
        return _support("unknown", cap_pressure, values)
    rank_one = top[0].chunk_id
    agreement = rank_one in vector_chunk_ids and (
        rank_one in trigram_chunk_ids or rank_one in exact_chunk_ids
    )
    values["rank_one_channel_agreement"] = agreement
    supported = not cap_pressure and mapped == len(top) and agreement
    return _support("supported" if supported else "insufficient", cap_pressure, values)


__all__ = [
    "PPRSeedSupportV1",
    "PreparedPPRSeedsV1",
    "summarize_direct_support",
    "summarize_extended_support",
]
