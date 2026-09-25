"""Versioned query-intent policy and within-pool score normalization."""

from __future__ import annotations

import math
import re

from apps.chat.services.rag_selection_types import SelectionProfile

# Includes the constant-score tolerance, because changing it changes the
# interpretation of downstream normalized relevance values.
POLICY_VERSION = "adaptive-evidence-v1-minmax-1e-9"
_PROFILE_VALUES = {
    "focused": (0.95, 0.05),
    "balanced": (0.90, 0.10),
    "breadth": (0.80, 0.15),
}
_BREADTH = re.compile(
    r"\b(?:compare|contrast|versus|vs\.?|synthesi[sz]e|synthesis)\b"
    r"|\b(?:literature|systematic)\s+review\b"
    r"|\bacross\s+(?:papers|studies|documents|trials)\b",
    re.IGNORECASE,
)
_FACTUAL = re.compile(r"\b(?:who|when|where|how\s+many|how\s+much)\b", re.IGNORECASE)
_MULTIPART = re.compile(
    r"\b(?:and|also)\s+(?:who|what|when|where|how|why)\b", re.IGNORECASE
)
_QUOTED_TITLE = re.compile(r'"[^"]*"|“[^”]*”|‘[^’]*’')


def choose_selection_profile(
    question: str, *, single_document: bool = False
) -> SelectionProfile:
    """Classify the resolved question; callers must pass question text, not a title."""
    cues = _QUOTED_TITLE.sub(" ", question)
    if _BREADTH.search(cues):
        name = "breadth"
    elif _MULTIPART.search(cues):
        name = "balanced"
    elif single_document or _FACTUAL.search(cues):
        name = "focused"
    else:
        name = "balanced"
    weight, allowance = _PROFILE_VALUES[name]
    return SelectionProfile(name, weight, allowance, POLICY_VERSION)


def normalize_relevance_scores(values: tuple[float, ...]) -> tuple[float, ...]:
    """Min/max normalize one comparable pool, without inferring confidence.

    Constant or nearly constant scores receive 0.5, so downstream ties stay
    stable and raw provider scales never become absolute selection thresholds.
    """
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("relevance scores must be finite numbers, not booleans")
    if not values:
        return ()
    low, high = min(values), max(values)
    spread = high - low
    if spread <= 1e-9 * max(1, abs(low), abs(high)):
        return (0.5,) * len(values)
    return tuple((value - low) / spread for value in values)
