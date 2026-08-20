# ruff: noqa: E501
# fmt: off
"""Closed, privacy-safe contracts for deterministic direct entity seeds."""
from __future__ import annotations

import re
from dataclasses import dataclass, fields
from enum import StrEnum
from math import fsum, isclose, isfinite
from typing import final

from lib.knowledge_graph.query_extractor.contracts import (
    MAX_QUERY_UTF8_BYTES,
    QueryEntitySpanV1,
)

_KEY = re.compile(r"[0-9a-f]{64}")
_MAX_SPANS = 128
_MAX_SEEDS = 64
_MAX_MATCHES = 128
_TIER_FACTORS: dict[DirectResolutionTier, float]
class DirectResolutionTier(StrEnum):
    IDENTIFIER = "identifier"
    NAME = "name"
    ALIAS = "alias"
    EMBEDDING = "embedding"
    @property
    def priority(self) -> int:
        return tuple(type(self)).index(self)
_TIER_FACTORS = {
    DirectResolutionTier.IDENTIFIER: 1.0,
    DirectResolutionTier.NAME: 0.95,
    DirectResolutionTier.ALIAS: 0.90,
    DirectResolutionTier.EMBEDDING: 0.80,
}
class DirectFailureReason(StrEnum):
    EXTRACTOR_TIMEOUT = "extractor_timeout"
    EXTRACTOR_AUTH = "extractor_auth"
    EXTRACTOR_PROVENANCE = "extractor_provenance"
    MIXED_ONTOLOGY = "mixed_ontology"
    DIRECT_SEED_INVALID = "direct_seed_invalid"
    DIRECT_NO_SEEDS = "direct_no_seeds"
    DIRECT_EMBEDDING_UNAVAILABLE = "direct_embedding_unavailable"
    DIRECT_TOPOLOGY_TIMEOUT = "direct_topology_timeout"
    DIRECT_TOPOLOGY_INVALID = "direct_topology_invalid"
    DIRECT_PPR_INVALID = "direct_ppr_invalid"
def _count(value: object, name: str, maximum: int = _MAX_SPANS) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} is outside its bound")
def _unit(value: object, name: str, *, positive: bool = False) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    lower = 0.0 < value if positive else 0.0 <= value
    if not isfinite(value) or not lower or value > 1.0:
        raise ValueError(f"{name} must be finite and in its unit interval")
def _key(value: object, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if _KEY.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex opaque key")
def _token(value: object, name: str, maximum: int = 128) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    if not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded canonical token")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} contains a forbidden control character")
@final
class DirectResolutionSpanInputV1:
    __slots__ = ("_span", "_text")
    def __init__(self, span: QueryEntitySpanV1, text: str) -> None:
        if type(span) is not QueryEntitySpanV1:
            raise TypeError("span must be an exact QueryEntitySpanV1")
        if type(text) is not str:
            raise TypeError("text must be an exact transient str")
        try:
            encoded = text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("text must be valid UTF-8") from error
        if len(encoded) > MAX_QUERY_UTF8_BYTES:
            raise ValueError("text exceeds its UTF-8 byte bound")
        if any(ord(character) < 32 or ord(character) == 127 for character in text):
            raise ValueError("text contains a forbidden C0/DEL control character")
        if len(text) != span.end - span.start:
            raise ValueError("text length must equal the span code-point width")
        object.__setattr__(self, "_span", span)
        object.__setattr__(self, "_text", text)
    @property
    def span(self) -> QueryEntitySpanV1:
        return self._span
    @property
    def text(self) -> str:
        return self._text
    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("transient direct-resolution input is immutable")
    def __delattr__(self, _name: str) -> None:
        raise AttributeError("transient direct-resolution input is immutable")
    def __repr__(self) -> str:
        return "<DirectResolutionSpanInputV1 redacted>"
    __str__ = __repr__
    def _blocked(self, *_args: object) -> object:
        raise TypeError("transient direct-resolution input is not serializable")
    __copy__ = __deepcopy__ = __reduce__ = __reduce_ex__ = _blocked
@final
@dataclass(frozen=True, slots=True)
class DirectEntityMatchV1:
    span_index: int
    entity_key: str
    component_key: str
    ontology_type: str
    tier: DirectResolutionTier
    extraction_confidence: float
    similarity: float
    match_weight: float
    def __post_init__(self) -> None:
        _count(self.span_index, "span_index", _MAX_SPANS - 1)
        _key(self.entity_key, "entity_key")
        _key(self.component_key, "component_key")
        _token(self.ontology_type, "ontology_type")
        if type(self.tier) is not DirectResolutionTier:
            raise TypeError("tier must be an exact DirectResolutionTier")
        _unit(self.extraction_confidence, "extraction_confidence")
        _unit(self.similarity, "similarity")
        _unit(self.match_weight, "match_weight", positive=True)
        if self.tier is not DirectResolutionTier.EMBEDDING and self.similarity != 1.0:
            raise ValueError("exact tiers require unit similarity")
        expected = (
            self.extraction_confidence
            * _TIER_FACTORS[self.tier]
            * (self.similarity if self.tier is DirectResolutionTier.EMBEDDING else 1.0)
        )
        if expected <= 0.0:
            raise ValueError("authoritative match_weight must be strictly positive")
        if self.match_weight != expected:
            raise ValueError("match_weight disagrees with its tier semantics")
@final
@dataclass(frozen=True, slots=True)
class ResolvedDirectSeedV1:
    component_key: str
    member_entity_keys: tuple[str, ...]
    mass: float
    def __post_init__(self) -> None:
        _key(self.component_key, "component_key")
        if type(self.member_entity_keys) is not tuple or not self.member_entity_keys:
            raise TypeError("member_entity_keys must be a nonempty exact tuple")
        if len(self.member_entity_keys) > _MAX_SEEDS:
            raise ValueError("member_entity_keys exceed the hard cap")
        for key in self.member_entity_keys:
            _key(key, "member_entity_key")
        if len(set(self.member_entity_keys)) != len(
            self.member_entity_keys
        ) or self.member_entity_keys != tuple(sorted(self.member_entity_keys)):
            raise ValueError("member_entity_keys must be unique and sorted")
        _unit(self.mass, "mass", positive=True)
@final
@dataclass(frozen=True, slots=True)
class DirectSeedAmbiguityV1:
    span_index: int
    tier: DirectResolutionTier
    component_count: int
    candidate_count: int
    def __post_init__(self) -> None:
        _count(self.span_index, "span_index", _MAX_SPANS - 1)
        if type(self.tier) is not DirectResolutionTier:
            raise TypeError("tier must be an exact DirectResolutionTier")
        _count(self.component_count, "component_count", _MAX_SEEDS)
        _count(self.candidate_count, "candidate_count", _MAX_MATCHES)
        if self.component_count < 2 or self.candidate_count < self.component_count:
            raise ValueError("ambiguity counts are incoherent")
@final
@dataclass(frozen=True, slots=True)
class DirectSeedDiagnosticsV1:
    input_span_count: int
    deduplicated_span_count: int
    resolved_span_count: int
    ambiguous_span_count: int
    unresolved_span_count: int
    embedding_attempt_count: int
    embedding_match_count: int
    def __post_init__(self) -> None:
        for field in fields(self):
            _count(getattr(self, field.name), field.name)
        if self.deduplicated_span_count > self.input_span_count:
            raise ValueError("deduplicated count exceeds input")
        if (
            self.resolved_span_count
            + self.ambiguous_span_count
            + self.unresolved_span_count
            != self.deduplicated_span_count
        ):
            raise ValueError("diagnostic span counts are incoherent")
        if self.embedding_match_count > self.embedding_attempt_count:
            raise ValueError("embedding diagnostic counts are incoherent")
        if self.embedding_attempt_count > self.deduplicated_span_count:
            raise ValueError("embedding attempts exceed deduplicated spans")
@final
@dataclass(frozen=True, slots=True)
class DirectSeedOutcomeV1:
    matches: tuple[DirectEntityMatchV1, ...]
    seeds: tuple[ResolvedDirectSeedV1, ...]
    ambiguities: tuple[DirectSeedAmbiguityV1, ...]
    diagnostics: DirectSeedDiagnosticsV1
    failure_reason: DirectFailureReason | None
    def __post_init__(self) -> None:
        _typed_rows(self.matches, DirectEntityMatchV1, "matches", _MAX_MATCHES)
        _typed_rows(self.seeds, ResolvedDirectSeedV1, "seeds", _MAX_SEEDS)
        _typed_rows(
            self.ambiguities,
            DirectSeedAmbiguityV1,
            "ambiguities",
            _MAX_SPANS,
        )
        if type(self.diagnostics) is not DirectSeedDiagnosticsV1:
            raise TypeError("diagnostics must be exact")
        if (
            self.failure_reason is not None
            and type(self.failure_reason) is not DirectFailureReason
        ):
            raise TypeError("failure_reason must be an exact DirectFailureReason")
        if self.failure_reason is not None and self.failure_reason not in (DirectFailureReason.EXTRACTOR_TIMEOUT, DirectFailureReason.EXTRACTOR_AUTH, DirectFailureReason.EXTRACTOR_PROVENANCE, DirectFailureReason.MIXED_ONTOLOGY, DirectFailureReason.DIRECT_SEED_INVALID, DirectFailureReason.DIRECT_NO_SEEDS, DirectFailureReason.DIRECT_EMBEDDING_UNAVAILABLE):
            raise ValueError("failure_reason is not valid at the direct-seed stage")
        match_keys = tuple(
            (row.span_index, row.tier.priority, row.component_key, row.entity_key)
            for row in self.matches
        )
        seed_keys = tuple((-row.mass, row.member_entity_keys[0]) for row in self.seeds)
        ambiguity_keys = tuple(row.span_index for row in self.ambiguities)
        _ordered_unique(match_keys, "matches")
        _ordered_unique(seed_keys, "seeds")
        _ordered_unique(ambiguity_keys, "ambiguities")
        if any(row.span_index >= self.diagnostics.deduplicated_span_count for row in (*self.matches, *self.ambiguities)):
            raise ValueError("outcome span index exceeds deduplicated span count")
        match_spans = {row.span_index for row in self.matches}
        if len(match_spans) != len(self.matches):
            raise ValueError("matches must retain one best component per span")
        if match_spans & {row.span_index for row in self.ambiguities}:
            raise ValueError("resolved and ambiguous spans must be disjoint")
        components = {row.component_key: row for row in self.seeds}
        match_components = {row.component_key for row in self.matches}
        member_keys = tuple(
            key for row in self.seeds for key in row.member_entity_keys
        )
        if len(set(member_keys)) != len(member_keys):
            raise ValueError("seed member entity keys must form a unique partition")
        if set(components) != match_components:
            raise ValueError("match/seed component closure is broken")
        if any(
            row.component_key not in components
            or row.entity_key not in components[row.component_key].member_entity_keys
            for row in self.matches
        ):
            raise ValueError("match/seed component closure is broken")
        if self.seeds and not isclose(
            fsum(seed.mass for seed in self.seeds), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("seed mass must normalize to one")
        if self.matches:
            total = fsum(row.match_weight for row in self.matches)
            for seed in self.seeds:
                expected = (
                    fsum(
                        row.match_weight
                        for row in self.matches
                        if row.component_key == seed.component_key
                    )
                    / total
                )
                if not isclose(seed.mass, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError("seed mass is not normalized component match mass")
        if self.diagnostics.resolved_span_count != len(self.matches) or (
            self.diagnostics.ambiguous_span_count != len(self.ambiguities)
        ):
            raise ValueError("diagnostics disagree with outcome rows")
        embedding_outcomes = sum(
            row.tier is DirectResolutionTier.EMBEDDING
            for row in (*self.matches, *self.ambiguities)
        )
        if self.diagnostics.embedding_match_count != embedding_outcomes:
            raise ValueError("embedding diagnostics disagree with tier outcomes")
        if self.failure_reason is None:
            if not self.matches or not self.seeds:
                raise ValueError("successful outcome requires resolved seeds")
        elif self.matches or self.seeds:
            raise ValueError("failure outcome must not expose partial seeds")
def _typed_rows(value: object, kind: type, name: str, cap: int) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(value) > cap:
        raise ValueError(f"{name} exceed the hard cap")
    if any(type(row) is not kind for row in value):
        raise TypeError(f"{name} must contain exact {kind.__name__} values")
def _ordered_unique(keys: tuple, name: str) -> None:
    if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
        raise ValueError(f"{name} must be unique and canonically sorted")
