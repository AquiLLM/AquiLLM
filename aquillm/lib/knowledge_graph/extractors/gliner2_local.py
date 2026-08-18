"""Lazy, process-local adapter for the pinned GLiNER2 runtime.

The optional provider is imported only while a graph worker loads its model.
Raw provider values are treated as untrusted candidate data: invalid values are
retained as provider-neutral diagnostics and never promoted to graph evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from threading import RLock
from typing import Any

from ..config import ExtractionSettings
from ..types import (
    EntityCandidate,
    ExtractionBatchResult,
    ExtractionDiagnostic,
    RelationCandidate,
)
from .base import OntologyDefinition

GLINER2_VERSION = "1.3.2"

_ModelKey = tuple[str, str, str, str, bool]
_MODEL_CACHE: dict[_ModelKey, Any] = {}
_MODEL_LOAD_LOCK = RLock()


class ExtractionBackendError(RuntimeError):
    """Raised when the optional provider cannot load or perform inference."""


def _model_key(settings: ExtractionSettings) -> _ModelKey:
    return (
        settings.model_id,
        settings.model_revision,
        str(settings.cache_dir),
        settings.device,
        settings.local_files_only,
    )


def _load_model(settings: ExtractionSettings) -> Any:
    """Load one pinned checkpoint once per process and effective configuration."""

    if len(settings.model_revision) != 40 or any(
        character not in "0123456789abcdefABCDEF"
        for character in settings.model_revision
    ):
        raise ExtractionBackendError(
            "GLiNER2 extraction requires an immutable 40-character revision"
        )
    key = _model_key(settings)
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    with _MODEL_LOAD_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None:
            return model
        try:
            import gliner2
            from gliner2 import GLiNER2
            from huggingface_hub import snapshot_download

            if getattr(gliner2, "__version__", None) != GLINER2_VERSION:
                raise ExtractionBackendError(
                    f"GLiNER2 runtime must be exactly {GLINER2_VERSION}"
                )
            local_snapshot_path = snapshot_download(
                repo_id=settings.model_id,
                revision=settings.model_revision,
                cache_dir=settings.cache_dir,
                local_files_only=settings.local_files_only,
            )
            model = GLiNER2.from_pretrained(
                local_snapshot_path,
                map_location=settings.device,
            )
        except ExtractionBackendError:
            raise
        except Exception as exc:
            raise ExtractionBackendError("GLiNER2 model load failed") from exc
        _MODEL_CACHE[key] = model
        return model


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


def _valid_span(
    text: str, surface: object, start: object, end: object
) -> bool:
    return (
        isinstance(surface, str)
        and bool(surface.strip())
        and type(start) is int
        and type(end) is int
        and 0 <= start < end <= len(text)
        and text[start:end] == surface
    )


def _normalize_entities(
    raw_result: object,
    *,
    text: str,
    input_index: int,
    known_types: frozenset[str],
) -> tuple[list[EntityCandidate], list[ExtractionDiagnostic]]:
    entities: list[EntityCandidate] = []
    diagnostics: list[ExtractionDiagnostic] = []
    if not isinstance(raw_result, Mapping):
        diagnostics.append(
            _diagnostic(
                "malformed_entity_output", "entity", input_index, reason="not_mapping"
            )
        )
        return entities, diagnostics
    if "entities" not in raw_result:
        diagnostics.append(
            _diagnostic(
                "missing_entity_output",
                "entity",
                input_index,
                reason="provider_section_absent",
            )
        )
        return entities, diagnostics
    raw_groups = raw_result["entities"]
    if not isinstance(raw_groups, Sequence) or isinstance(
        raw_groups, (str, bytes)
    ):
        diagnostics.append(
            _diagnostic(
                "malformed_entity_output",
                "entity",
                input_index,
                reason="entities_not_sequence",
            )
        )
        return entities, diagnostics
    if not raw_groups:
        return entities, diagnostics
    if len(raw_groups) != 1 or not isinstance(raw_groups[0], Mapping):
        diagnostics.append(
            _diagnostic(
                "malformed_entity_output",
                "entity",
                input_index,
                reason="invalid_entity_group",
            )
        )
        return entities, diagnostics
    grouped = raw_groups[0]

    for entity_type, candidates in grouped.items():
        if not isinstance(candidates, Sequence) or isinstance(
            candidates, (str, bytes)
        ):
            diagnostics.append(
                _diagnostic(
                    "malformed_entity_output",
                    "entity",
                    input_index,
                    entity_type=str(entity_type),
                    reason="candidates_not_sequence",
                )
            )
            continue
        for raw_candidate in candidates:
            if not isinstance(entity_type, str) or entity_type not in known_types:
                diagnostics.append(
                    _diagnostic(
                        "unknown_entity_type",
                        "entity",
                        input_index,
                        entity_type=str(entity_type),
                    )
                )
                continue
            if not isinstance(raw_candidate, Mapping):
                diagnostics.append(
                    _diagnostic(
                        "malformed_entity_output",
                        "entity",
                        input_index,
                        entity_type=entity_type,
                        reason="candidate_not_mapping",
                    )
                )
                continue
            surface = raw_candidate.get("text")
            confidence = raw_candidate.get("confidence")
            start = raw_candidate.get("start")
            end = raw_candidate.get("end")
            if not _valid_confidence(confidence):
                diagnostics.append(
                    _diagnostic(
                        "invalid_entity_confidence",
                        "entity",
                        input_index,
                        entity_type=entity_type,
                        surface=surface if isinstance(surface, str) else None,
                    )
                )
                continue
            if not _valid_span(text, surface, start, end):
                diagnostics.append(
                    _diagnostic(
                        "malformed_entity_span",
                        "entity",
                        input_index,
                        entity_type=entity_type,
                        surface=surface if isinstance(surface, str) else None,
                        start=start if type(start) is int else None,
                        end=end if type(end) is int else None,
                    )
                )
                continue
            entities.append(
                EntityCandidate(
                    entity_type=entity_type,
                    text=surface,
                    start=start,
                    end=end,
                    confidence=float(confidence),
                )
            )
    return entities, diagnostics


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
        endpoint_mapping = (
            raw_endpoint if isinstance(raw_endpoint, Mapping) else {}
        )
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
    relation_type: str,
    relation_definition: object,
    raw_candidate: Mapping[object, object],
    entities: Sequence[EntityCandidate],
    text: str,
    input_index: int,
) -> tuple[tuple[str, int, int, float] | None, ExtractionDiagnostic | None]:
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

    allowed = _allowed_types(relation_definition, endpoint)
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
            return (surface, start, end, float(confidence)), None
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

    compatible = [
        entity for entity in surface_matches if entity.entity_type in allowed
    ]
    if len(compatible) == 1:
        mention = compatible[0]
        return (
            surface,
            mention.start,
            mention.end,
            float(confidence),
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


def _normalize_relations(
    raw_result: object,
    *,
    text: str,
    input_index: int,
    ontology_relations: Mapping[str, object],
    entities: Sequence[EntityCandidate],
) -> tuple[list[RelationCandidate], list[ExtractionDiagnostic]]:
    relations: list[RelationCandidate] = []
    diagnostics: list[ExtractionDiagnostic] = []
    if not isinstance(raw_result, Mapping):
        diagnostics.append(
            _diagnostic(
                "malformed_relation_output",
                "relation",
                input_index,
                reason="not_mapping",
            )
        )
        return relations, diagnostics
    for expected_relation_type in ontology_relations:
        if expected_relation_type not in raw_result:
            diagnostics.append(
                _diagnostic(
                    "missing_relation_output",
                    "relation",
                    input_index,
                    relation_type=expected_relation_type,
                    reason="provider_section_absent",
                )
            )

    for relation_type, candidates in raw_result.items():
        if relation_type == "entities":
            continue
        if not isinstance(candidates, Sequence) or isinstance(
            candidates, (str, bytes)
        ):
            diagnostics.append(
                _diagnostic(
                    "malformed_relation_output",
                    "relation",
                    input_index,
                    relation_type=str(relation_type),
                    reason="candidates_not_sequence",
                )
            )
            continue
        for raw_candidate in candidates:
            normalized_relation_type = str(relation_type)
            if (
                not isinstance(relation_type, str)
                or relation_type not in ontology_relations
            ):
                if isinstance(raw_candidate, Mapping):
                    diagnostics.append(
                        _relation_error(
                            "unknown_relation_type",
                            input_index=input_index,
                            relation_type=normalized_relation_type,
                            raw_candidate=raw_candidate,
                            endpoint=None,
                        )
                    )
                else:
                    diagnostics.append(
                        _diagnostic(
                            "unknown_relation_type",
                            "relation",
                            input_index,
                            relation_type=normalized_relation_type,
                        )
                    )
                continue
            if not isinstance(raw_candidate, Mapping):
                diagnostics.append(
                    _diagnostic(
                        "malformed_relation_output",
                        "relation",
                        input_index,
                        relation_type=relation_type,
                        reason="candidate_not_mapping",
                    )
                )
                continue
            definition = ontology_relations[relation_type]
            head, head_error = _resolve_endpoint(
                raw_candidate.get("head"),
                endpoint="head",
                relation_type=relation_type,
                relation_definition=definition,
                raw_candidate=raw_candidate,
                entities=entities,
                text=text,
                input_index=input_index,
            )
            if head_error is not None:
                diagnostics.append(head_error)
                continue
            tail, tail_error = _resolve_endpoint(
                raw_candidate.get("tail"),
                endpoint="tail",
                relation_type=relation_type,
                relation_definition=definition,
                raw_candidate=raw_candidate,
                entities=entities,
                text=text,
                input_index=input_index,
            )
            if tail_error is not None:
                diagnostics.append(tail_error)
                continue
            assert head is not None and tail is not None
            relations.append(
                RelationCandidate(
                    relation_type=relation_type,
                    head_text=head[0],
                    tail_text=tail[0],
                    head_start=head[1],
                    head_end=head[2],
                    tail_start=tail[1],
                    tail_end=tail[2],
                    confidence=min(head[3], tail[3]),
                )
            )
    return relations, diagnostics


def _require_batch_results(
    value: object, expected_count: int, candidate_kind: str
) -> Sequence[object]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != expected_count
    ):
        raise ExtractionBackendError(
            f"GLiNER2 returned an invalid {candidate_kind} result batch"
        )
    return value


class GLiNER2LocalBackend:
    """Provider-neutral extraction backed by a pinned local GLiNER2 model."""

    def __init__(self, *, settings: ExtractionSettings) -> None:
        self._settings = settings

    def extract_batch(
        self,
        texts: tuple[str, ...],
        *,
        ontology: OntologyDefinition,
    ) -> tuple[ExtractionBatchResult, ...]:
        if not texts:
            return ()

        model = _load_model(self._settings)
        entity_definitions = {
            name: (
                description
                if isinstance(
                    description := _definition_value(definition, "description"),
                    str,
                )
                and description.strip()
                else name
            )
            for name, definition in ontology.entity_types.items()
        }
        relation_definitions = {
            name: (
                description
                if isinstance(
                    description := _definition_value(definition, "description"),
                    str,
                )
                and description.strip()
                else name
            )
            for name, definition in ontology.relations.items()
        }
        inference_options = {
            "batch_size": self._settings.batch_size,
            "format_results": False,
            "include_confidence": True,
            "include_spans": True,
        }
        try:
            schema = (
                model.create_schema()
                .entities(entity_definitions)
                .relations(relation_definitions)
            )
            raw_results = model.batch_extract(
                list(texts), schema, **inference_options
            )
        except Exception as exc:
            raise ExtractionBackendError("GLiNER2 inference failed") from exc

        batches = _require_batch_results(raw_results, len(texts), "composite")
        results: list[ExtractionBatchResult] = []
        known_types = frozenset(ontology.entity_types)
        ontology_relations = ontology.relations
        for input_index, text in enumerate(texts):
            entities, entity_diagnostics = _normalize_entities(
                batches[input_index],
                text=text,
                input_index=input_index,
                known_types=known_types,
            )
            relations, relation_diagnostics = _normalize_relations(
                batches[input_index],
                text=text,
                input_index=input_index,
                ontology_relations=ontology_relations,
                entities=entities,
            )
            results.append(
                ExtractionBatchResult(
                    entities=tuple(entities),
                    relations=tuple(relations),
                    diagnostics=tuple(entity_diagnostics + relation_diagnostics),
                )
            )
        return tuple(results)


__all__ = ["ExtractionBackendError", "GLiNER2LocalBackend"]
