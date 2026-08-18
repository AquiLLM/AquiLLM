"""Provider-neutral scoring primitives for collection entity resolution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite, sqrt
from struct import error as struct_error
from struct import pack, unpack

EMBEDDING_DIMENSIONS = 1024


class ResolutionOutcome(StrEnum):
    """Review disposition for one possible identity edge."""

    AUTOMATIC = "automatic"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


class ResolutionTier(StrEnum):
    """The strongest resolver tier contributing to a pair decision."""

    STABLE_IDENTIFIER = "stable_identifier"
    EXACT_LABEL_OR_ALIAS = "exact_label_or_alias"
    EMBEDDING = "embedding"
    NEIGHBORHOOD_AGREEMENT = "neighborhood_agreement"


def _unit_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class ResolutionThresholds:
    """Independent thresholds for retrieval, review, and identity promotion."""

    automatic: float = 0.93
    candidate: float = 0.76
    retrieval_similarity: float = 0.70

    def __post_init__(self) -> None:
        automatic = _unit_float(self.automatic, "automatic threshold")
        candidate = _unit_float(self.candidate, "candidate threshold")
        retrieval = _unit_float(
            self.retrieval_similarity, "retrieval similarity threshold"
        )
        if automatic <= candidate:
            raise ValueError(
                "automatic identity threshold must be stricter than candidate threshold"
            )
        if automatic <= retrieval:
            raise ValueError(
                "automatic identity threshold must be stricter than retrieval "
                "similarity threshold"
            )
        object.__setattr__(self, "automatic", automatic)
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "retrieval_similarity", retrieval)


@dataclass(frozen=True, slots=True)
class ResolutionScore:
    """Transparent components used to classify one identity candidate."""

    composite: float
    embedding_similarity: float | None
    neighborhood_agreement: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "composite", _unit_float(self.composite, "composite score")
        )
        for name in ("embedding_similarity", "neighborhood_agreement"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _unit_float(value, name))


def classify_resolution_score(
    score: float,
    thresholds: ResolutionThresholds,
) -> ResolutionOutcome:
    """Classify a score without conflating retrieval and merge thresholds."""

    value = _unit_float(score, "resolution score")
    if type(thresholds) is not ResolutionThresholds:
        raise ValueError("thresholds must be an exact ResolutionThresholds value")
    thresholds.__post_init__()
    if value >= thresholds.automatic:
        return ResolutionOutcome.AUTOMATIC
    if value >= thresholds.candidate:
        return ResolutionOutcome.CANDIDATE
    return ResolutionOutcome.REJECTED


def validate_embedding(
    value: object,
    *,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> tuple[float, ...]:
    """Return an immutable finite vector with the persisted KG dimensionality."""

    if type(dimensions) is not int or dimensions <= 0:
        raise ValueError("embedding dimensions must be a positive integer")
    if not isinstance(value, (list, tuple)) or len(value) != dimensions:
        raise ValueError(f"embedding must contain exactly {dimensions} dimensions")
    vector: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError("embedding components must be finite numbers")
        number = float(component)
        if not isfinite(number):
            raise ValueError("embedding components must be finite numbers")
        try:
            quantized = unpack("!f", pack("!f", number))[0]
        except (OverflowError, struct_error) as exc:
            raise ValueError("embedding components must fit finite float32") from exc
        if not isfinite(quantized):
            raise ValueError("embedding components must remain finite as float32")
        vector.append(quantized)
    return tuple(vector)


def cosine_similarity(
    left: Iterable[float],
    right: Iterable[float],
) -> float:
    """Return nonnegative cosine similarity for two validated KG embeddings."""

    left_vector = validate_embedding(left)
    right_vector = validate_embedding(right)
    dot = sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
    left_norm = sqrt(sum(component * component for component in left_vector))
    right_norm = sqrt(sum(component * component for component in right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    # Negative semantic similarity is not useful as identity evidence.
    return min(1.0, max(0.0, dot / (left_norm * right_norm)))


def combine_resolution_scores(
    *,
    embedding_similarity: float,
    neighborhood_agreement: float,
    embedding_weight: float,
    neighborhood_weight: float,
) -> ResolutionScore:
    """Blend independently recorded evidence with explicit normalized weights."""

    embedding = _unit_float(embedding_similarity, "embedding similarity")
    neighborhood = _unit_float(neighborhood_agreement, "neighborhood agreement")
    weights: list[float] = []
    for value, label in (
        (embedding_weight, "embedding weight"),
        (neighborhood_weight, "neighborhood weight"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite nonnegative number")
        number = float(value)
        if not isfinite(number) or number < 0:
            raise ValueError(f"{label} must be a finite nonnegative number")
        weights.append(number)
    total = sum(weights)
    if total <= 0:
        raise ValueError("resolution score weights must have a positive sum")
    composite = (embedding * weights[0] + neighborhood * weights[1]) / total
    return ResolutionScore(
        composite=composite,
        embedding_similarity=embedding,
        neighborhood_agreement=neighborhood,
    )


__all__ = [
    "EMBEDDING_DIMENSIONS",
    "ResolutionOutcome",
    "ResolutionScore",
    "ResolutionThresholds",
    "ResolutionTier",
    "classify_resolution_score",
    "combine_resolution_scores",
    "cosine_similarity",
    "validate_embedding",
]
