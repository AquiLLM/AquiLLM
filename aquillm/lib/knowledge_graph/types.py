"""Immutable, provider-neutral contracts for graph extraction."""

from dataclasses import dataclass
from math import isfinite


DiagnosticScalar = str | int | float | bool | None


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _require_nonnegative_int(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


def _require_nonempty_span(start: object, end: object, prefix: str = "") -> None:
    start_name = f"{prefix}start"
    end_name = f"{prefix}end"
    _require_nonnegative_int(start, start_name)
    if type(end) is not int or end <= start:
        raise ValueError(f"{end_name} must be greater than {start_name}")


def _require_confidence(value: object) -> None:
    if type(value) not in (int, float):
        raise ValueError("confidence must be a finite number in [0, 1]")
    if type(value) is float and not isfinite(value):
        raise ValueError("confidence must be a finite number in [0, 1]")
    if not 0 <= value <= 1:
        raise ValueError("confidence must be a finite number in [0, 1]")


def _is_diagnostic_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and isfinite(value)


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    """An entity proposed by an extractor, with a half-open text span."""

    entity_type: str
    text: str
    start: int
    end: int
    confidence: float

    def __post_init__(self) -> None:
        _require_nonempty_string(self.entity_type, "entity_type")
        _require_nonempty_string(self.text, "text")
        _require_nonempty_span(self.start, self.end)
        _require_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class RelationCandidate:
    """A relation proposed by an extractor, with half-open endpoint spans."""

    relation_type: str
    head_text: str
    tail_text: str
    head_start: int
    head_end: int
    tail_start: int
    tail_end: int
    confidence: float

    def __post_init__(self) -> None:
        _require_nonempty_string(self.relation_type, "relation_type")
        _require_nonempty_string(self.head_text, "head_text")
        _require_nonempty_string(self.tail_text, "tail_text")
        _require_nonempty_span(self.head_start, self.head_end, "head_")
        _require_nonempty_span(self.tail_start, self.tail_end, "tail_")
        _require_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class ExtractionDiagnostic:
    """Stable, provider-neutral evidence for rejected extractor output."""

    code: str
    candidate_kind: str
    input_index: int
    details: tuple[tuple[str, DiagnosticScalar], ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.code, "code")
        _require_nonempty_string(self.candidate_kind, "candidate_kind")
        _require_nonnegative_int(self.input_index, "input_index")
        if not isinstance(self.details, tuple):
            raise ValueError("details must be a tuple")
        for detail in self.details:
            if not isinstance(detail, tuple) or len(detail) != 2:
                raise ValueError("details must contain (key, scalar value) tuples")
            key, value = detail
            _require_nonempty_string(key, "details key")
            if not _is_diagnostic_scalar(value):
                raise ValueError("details values must be scalar")


@dataclass(frozen=True, slots=True)
class ExtractionBatchResult:
    """Extractor output and immutable evidence for rejected candidates."""

    entities: tuple[EntityCandidate, ...]
    relations: tuple[RelationCandidate, ...]
    diagnostics: tuple[ExtractionDiagnostic, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entities, tuple):
            raise ValueError("entities must be a tuple")
        if not isinstance(self.relations, tuple):
            raise ValueError("relations must be a tuple")
        if not isinstance(self.diagnostics, tuple):
            raise ValueError("diagnostics must be a tuple")
        if not all(isinstance(entity, EntityCandidate) for entity in self.entities):
            raise ValueError("entities must contain EntityCandidate values")
        if not all(isinstance(relation, RelationCandidate) for relation in self.relations):
            raise ValueError("relations must contain RelationCandidate values")
        if not all(isinstance(diagnostic, ExtractionDiagnostic) for diagnostic in self.diagnostics):
            raise ValueError("diagnostics must contain ExtractionDiagnostic values")
