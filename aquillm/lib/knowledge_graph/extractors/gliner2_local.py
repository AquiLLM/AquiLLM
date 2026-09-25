"""Lazy, process-local adapter for the pinned GLiNER2 runtime.

The optional provider is imported only while a graph worker loads its model.
Raw provider values are treated as untrusted candidate data: invalid values are
retained as provider-neutral diagnostics and never promoted to graph evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Any

from ..config import ExtractionSettings
from ..type_names import TypeNameValidationError, validate_type_name
from ..types import (
    EntityCandidate,
    ExtractionBatchResult,
    ExtractionDiagnostic,
    RelationCandidate,
)
from .base import OntologyDefinition
from .gliner2_endpoints import (
    _definition_value,
    _diagnostic,
    _relation_error,
    _resolve_endpoint,
    _valid_confidence,
    _valid_span,
)

GLINER2_VERSION = "1.3.2"

_ModelKey = tuple[str, str, str, str, bool]
_MODEL_CACHE: dict[_ModelKey, Any] = {}
_MODEL_LOAD_LOCK = RLock()


class ExtractionBackendError(RuntimeError):
    """Raised when the optional provider cannot load or perform inference."""


def _validate_ontology_type_names(ontology: OntologyDefinition) -> None:
    try:
        for name in ontology.entity_types:
            validate_type_name(name, "ontology entity type name")
        for name in ontology.relations:
            validate_type_name(name, "ontology relation type name")
    except TypeNameValidationError as exc:
        raise ExtractionBackendError(str(exc)) from exc


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
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
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
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
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
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
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
            endpoint_orientations = [("head", "tail")]
            if _definition_value(definition, "direction") == "undirected":
                endpoint_orientations.append(("tail", "head"))
            resolved_pairs = {}
            ambiguous_error = None
            best_error: ExtractionDiagnostic | None = None
            best_resolved_endpoint_count = -1
            for head_role, tail_role in endpoint_orientations:
                head, head_error = _resolve_endpoint(
                    raw_candidate.get("head"),
                    endpoint="head",
                    allowed_endpoint=head_role,
                    relation_type=relation_type,
                    relation_definition=definition,
                    raw_candidate=raw_candidate,
                    entities=entities,
                    text=text,
                    input_index=input_index,
                )
                tail, tail_error = _resolve_endpoint(
                    raw_candidate.get("tail"),
                    endpoint="tail",
                    allowed_endpoint=tail_role,
                    relation_type=relation_type,
                    relation_definition=definition,
                    raw_candidate=raw_candidate,
                    entities=entities,
                    text=text,
                    input_index=input_index,
                )
                if head_error is not None or tail_error is not None:
                    errors = (head_error, tail_error)
                    if all(
                        error is None or error.code == "ambiguous_relation_endpoint"
                        for error in errors
                    ):
                        ambiguous_error = next(
                            error for error in errors if error is not None
                        )
                    resolved_count = int(head_error is None)
                    if best_resolved_endpoint_count < resolved_count:
                        best_error = head_error or tail_error
                        best_resolved_endpoint_count = resolved_count
                    continue
                assert head is not None and tail is not None
                # Retain type identity: equal surface spans can denote distinct
                # typed entities, while overlapping role rules can yield the
                # same exact pair in both orientations.
                pair_key = (head[:3] + head[4:], tail[:3] + tail[4:])
                if len(endpoint_orientations) == 2:
                    pair_key = tuple(sorted(pair_key))
                resolved_pairs.setdefault(pair_key, (head, tail))
            if ambiguous_error is not None or len(resolved_pairs) > 1:
                diagnostics.append(
                    ambiguous_error
                    or _relation_error(
                        "ambiguous_relation_endpoint",
                        input_index=input_index,
                        relation_type=relation_type,
                        raw_candidate=raw_candidate,
                        endpoint="head"
                        if len({pair[0] for pair in resolved_pairs}) > 1
                        else "tail",
                    )
                )
                continue
            if not resolved_pairs:
                assert best_error is not None
                diagnostics.append(best_error)
                continue
            head, tail = next(iter(resolved_pairs.values()))
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
        return self._extract_batch(texts, ontology=ontology, include_relations=True)

    def extract_entities_batch(
        self,
        texts: tuple[str, ...],
        *,
        ontology: OntologyDefinition,
    ) -> tuple[ExtractionBatchResult, ...]:
        """Extract query anchors without running document relation inference."""
        return self._extract_batch(texts, ontology=ontology, include_relations=False)

    def _extract_batch(
        self,
        texts: tuple[str, ...],
        *,
        ontology: OntologyDefinition,
        include_relations: bool,
    ) -> tuple[ExtractionBatchResult, ...]:
        if not texts:
            return ()

        _validate_ontology_type_names(ontology)
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
        inference_options = {
            "batch_size": self._settings.batch_size,
            "format_results": False,
            "include_confidence": True,
            "include_spans": True,
        }
        try:
            schema = model.create_schema().entities(entity_definitions)
            if include_relations:
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
                schema = schema.relations(relation_definitions)
            raw_results = model.batch_extract(list(texts), schema, **inference_options)
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
            relations, relation_diagnostics = (
                _normalize_relations(
                    batches[input_index],
                    text=text,
                    input_index=input_index,
                    ontology_relations=ontology_relations,
                    entities=entities,
                )
                if include_relations
                else ([], [])
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
