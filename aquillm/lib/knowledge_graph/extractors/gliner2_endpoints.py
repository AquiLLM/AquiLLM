"""Provider-neutral validation and grounding of relation endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from ..types import EntityCandidate, ExtractionDiagnostic


def _diagnostic(
    code: str,
    candidate_kind: str,
    input_index: int,
    **details: str | int | float | bool | None,
) -> ExtractionDiagnostic:
    return ExtractionDiagnostic(
        code=code,
        candidate_kind=candidate_kind,
        input_index=input_index,
        details=tuple(sorted(details.items())),
    )


def _valid_confidence(value: object) -> bool:
    if type(value) is int:
        return 0 <= value <= 1
    if type(value) is float:
        return isfinite(value) and 0.0 <= value <= 1.0
    return False


def _valid_span(text: str, surface: object, start: object, end: object) -> bool:
    return (
        isinstance(surface, str)
        and bool(surface.strip())
        and type(start) is int
        and type(end) is int
        and 0 <= start < end <= len(text)
        and text[start:end] == surface
    )


def _definition_value(definition: object, name: str) -> object:
    if isinstance(definition, Mapping):
        return definition.get(name)
    return getattr(definition, name, None)


def _allowed_types(definition: object, endpoint: str) -> frozenset[str]:
    value = _definition_value(definition, f"allowed_{endpoint}_types")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return frozenset()
    return frozenset(item for item in value if isinstance(item, str))


def _safe_diagnostic_value(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    return None


def _relation_error(
    code: str,
    *,
    input_index: int,
    relation_type: str,
    raw_candidate: Mapping[object, object],
    endpoint: str | None,
) -> ExtractionDiagnostic:
    details: dict[str, str | int | float | bool | None] = {
        "relation_type": relation_type,
        "endpoint": endpoint,
        "endpoint_text": None,
    }
    for side in ("head", "tail"):
        raw_endpoint = raw_candidate.get(side)
        endpoint_mapping = raw_endpoint if isinstance(raw_endpoint, Mapping) else {}
        for field in ("text", "start", "end", "confidence"):
            details[f"{side}_{field}"] = _safe_diagnostic_value(
                endpoint_mapping.get(field)
            )
    if endpoint in ("head", "tail"):
        details["endpoint_text"] = details[f"{endpoint}_text"]
    return _diagnostic(
        code,
        "relation",
        input_index,
        **details,
    )


def _resolve_endpoint(
    raw_endpoint: object,
    *,
    endpoint: str,
    allowed_endpoint: str,
    relation_type: str,
    relation_definition: object,
    raw_candidate: Mapping[object, object],
    entities: Sequence[EntityCandidate],
    text: str,
    input_index: int,
) -> tuple[tuple[str, int, int, float, str] | None, ExtractionDiagnostic | None]:
    if not isinstance(raw_endpoint, Mapping):
        return None, _relation_error(
            "malformed_relation_endpoint",
            input_index=input_index,
            relation_type=relation_type,
            raw_candidate=raw_candidate,
            endpoint=endpoint,
        )
    surface = raw_endpoint.get("text")
    confidence = raw_endpoint.get("confidence")
    if not _valid_confidence(confidence):
        return None, _relation_error(
            "invalid_relation_confidence",
            input_index=input_index,
            relation_type=relation_type,
            raw_candidate=raw_candidate,
            endpoint=endpoint,
        )
    if not isinstance(surface, str) or not surface.strip():
        return None, _relation_error(
            "malformed_relation_endpoint",
            input_index=input_index,
            relation_type=relation_type,
            raw_candidate=raw_candidate,
            endpoint=endpoint,
        )

    allowed = _allowed_types(relation_definition, allowed_endpoint)
    surface_matches = [entity for entity in entities if entity.text == surface]
    has_start = "start" in raw_endpoint
    has_end = "end" in raw_endpoint
    if has_start or has_end:
        start = raw_endpoint.get("start")
        end = raw_endpoint.get("end")
        if not (has_start and has_end and _valid_span(text, surface, start, end)):
            return None, _relation_error(
                "malformed_relation_span",
                input_index=input_index,
                relation_type=relation_type,
                raw_candidate=raw_candidate,
                endpoint=endpoint,
            )
        spanned_matches = [
            entity
            for entity in surface_matches
            if entity.start == start and entity.end == end
        ]
        compatible = [
            entity for entity in spanned_matches if entity.entity_type in allowed
        ]
        if len(compatible) == 1:
            return (
                surface,
                start,
                end,
                float(confidence),
                compatible[0].entity_type,
            ), None
        if len(compatible) > 1:
            return None, _relation_error(
                "ambiguous_relation_endpoint",
                input_index=input_index,
                relation_type=relation_type,
                raw_candidate=raw_candidate,
                endpoint=endpoint,
            )
        code = (
            "disallowed_relation_endpoint"
            if spanned_matches
            else "unresolved_relation_endpoint"
        )
        return None, _relation_error(
            code,
            input_index=input_index,
            relation_type=relation_type,
            raw_candidate=raw_candidate,
            endpoint=endpoint,
        )

    compatible = [entity for entity in surface_matches if entity.entity_type in allowed]
    if len(compatible) == 1:
        mention = compatible[0]
        return (
            surface,
            mention.start,
            mention.end,
            float(confidence),
            mention.entity_type,
        ), None
    if len(compatible) > 1:
        return None, _relation_error(
            "ambiguous_relation_endpoint",
            input_index=input_index,
            relation_type=relation_type,
            raw_candidate=raw_candidate,
            endpoint=endpoint,
        )
    code = (
        "disallowed_relation_endpoint"
        if surface_matches
        else "unresolved_relation_endpoint"
    )
    return None, _relation_error(
        code,
        input_index=input_index,
        relation_type=relation_type,
        raw_candidate=raw_candidate,
        endpoint=endpoint,
    )
