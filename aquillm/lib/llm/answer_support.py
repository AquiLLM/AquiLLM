"""Request-only support decisions; relevance and citation validity are not proof."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AnswerSupport:
    state: Literal["complete", "partial", "insufficient", "unknown"]
    requested_aspects: tuple[str, ...]
    supported_aspects: tuple[str, ...]
    unresolved_aspects: tuple[str, ...]
    evidence_digest: str
    scope_digest: str
    assessment_basis: Literal["deterministic", "validated_semantic", "unvalidated"]
    calibration_id: str | None = None
    exhaustive: bool = False
