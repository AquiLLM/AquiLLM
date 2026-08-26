"""Chunk-bounded document extraction and immutable mention persistence."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import islice
from time import perf_counter
from typing import TYPE_CHECKING

import structlog

from lib.knowledge_graph.types import RelationCandidate

from ..resolution import DOCUMENT_RESOLVER_VERSION
from .windows import (
    ExtractionWindow,
    MappedEntityEvidence,
    batch_extraction_windows,
    deduplicate_mapped_entities,
    map_entity_candidate,
    sanitize_graph_source_text,
)

if TYPE_CHECKING:
    from lib.knowledge_graph.extractors.base import (
        ExtractionBackend,
        OntologyDefinition,
    )


_FATAL_STRUCTURAL_DIAGNOSTICS = frozenset(
    {"missing_entity_output", "missing_relation_output"}
)
DOCUMENT_EXTRACTION_V1_MAX_CHUNKS = 10_000
DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS = 10_000_000
DOCUMENT_EXTRACTION_V1_MAX_ENTITIES = 512
DOCUMENT_EXTRACTION_V1_MAX_RELATIONS = 4_096
DOCUMENT_EXTRACTION_V1_MAX_RAW_ENTITY_OBSERVATIONS = 4_096
DOCUMENT_EXTRACTION_V1_MAX_RAW_RELATION_OBSERVATIONS = 32_768
_QUERY_ITERATOR_BATCH_SIZE = 1_000
logger = structlog.stdlib.get_logger(__name__)


class StructuralExtractionError(RuntimeError):
    """Raised when a provider omits a required per-window output section."""


class StaleSourceError(RuntimeError):
    """Raised before writes when the requested source snapshot is stale."""


class MidflightSourceChangedError(StaleSourceError):
    """Raised when document text or ordered chunks change during inference."""


class OntologySnapshotChangedError(StaleSourceError):
    """Raised when a selected persisted ontology is no longer reproducible."""


class DocumentResolutionError(LookupError):
    """Raised when a UUID does not resolve to exactly one concrete document."""


class ExtractionInProgressError(RuntimeError):
    """Raised when the same immutable build identity is already in progress."""


@dataclass(frozen=True, slots=True)
class RelationObservation:
    chunk_id: int
    confidence: float
    modality: str
    head_local_start: int
    head_local_end: int
    tail_local_start: int
    tail_local_end: int


@dataclass(frozen=True, slots=True)
class MappedRelationEvidence:
    document_id: object
    chunk_id: int
    relation_type: str
    head_identity: tuple[object, ...]
    tail_identity: tuple[object, ...]
    confidence: float
    observations: tuple[RelationObservation, ...]

    @property
    def identity_key(self) -> tuple[object, ...]:
        return (self.relation_type, self.head_identity, self.tail_identity)


@dataclass(frozen=True, slots=True)
class ExtractedDocumentEvidence:
    entities: tuple[MappedEntityEvidence, ...]
    relations: tuple[MappedRelationEvidence, ...]
    diagnostic_counts: dict[str, int]
    window_count: int
    batch_count: int


def serialize_entity_observations(
    entity: MappedEntityEvidence,
) -> list[dict[str, str | int | float | bool | None]]:
    """Return immutable observation provenance containing JSON scalar values only."""

    return [
        {
            "chunk_id": observation.chunk_id,
            "confidence": observation.confidence,
            "modality": observation.modality,
            "position_basis": observation.position_basis,
            "start": observation.start,
            "end": observation.end,
            "local_start": observation.local_start,
            "local_end": observation.local_end,
            "content_object_type_id": observation.content_object_type_id,
            "content_object_id": (
                str(observation.content_object_id)
                if observation.content_object_id is not None
                else None
            ),
        }
        for observation in entity.observations
    ]


def _definition_value(definition: object, field_name: str) -> object:
    if isinstance(definition, dict):
        return definition.get(field_name)
    return getattr(definition, field_name, None)


def _allowed_endpoint_types(
    ontology: OntologyDefinition,
    relation_type: str,
    endpoint: str,
) -> frozenset[str]:
    relation_definition = ontology.relations.get(relation_type)
    if relation_definition is None:
        return frozenset()
    raw_types = _definition_value(relation_definition, f"allowed_{endpoint}_types")
    if not isinstance(raw_types, (tuple, list)):
        return frozenset()
    return frozenset(value for value in raw_types if isinstance(value, str))


def _endpoint_entity(
    relation: RelationCandidate,
    endpoint: str,
    mapped_entities: tuple[MappedEntityEvidence, ...],
    ontology: OntologyDefinition,
) -> MappedEntityEvidence | None:
    surface = getattr(relation, f"{endpoint}_text")
    start = getattr(relation, f"{endpoint}_start")
    end = getattr(relation, f"{endpoint}_end")
    allowed = _allowed_endpoint_types(ontology, relation.relation_type, endpoint)
    matches = [
        entity
        for entity in mapped_entities
        if entity.observations[0].local_start == start
        and entity.observations[0].local_end == end
        and entity.raw_text == surface
        and entity.entity_type in allowed
    ]
    return matches[0] if len(matches) == 1 else None


def _deduplicate_relations(
    relations: list[MappedRelationEvidence],
) -> tuple[MappedRelationEvidence, ...]:
    deduplicated: dict[tuple[object, ...], MappedRelationEvidence] = {}
    for relation in relations:
        existing = deduplicated.get(relation.identity_key)
        if existing is None:
            deduplicated[relation.identity_key] = relation
            continue
        representative = (
            relation if relation.confidence > existing.confidence else existing
        )
        observations = tuple(
            sorted(
                (*existing.observations, *relation.observations),
                key=lambda item: (
                    item.chunk_id,
                    item.head_local_start,
                    item.tail_local_start,
                ),
            )
        )
        deduplicated[relation.identity_key] = replace(
            representative,
            observations=observations,
        )
    return tuple(
        sorted(
            deduplicated.values(),
            key=lambda item: (
                item.relation_type,
                repr(item.head_identity),
                repr(item.tail_identity),
            ),
        )
    )


def collect_document_evidence(
    windows: tuple[ExtractionWindow, ...],
    *,
    full_text: str,
    backend: ExtractionBackend,
    ontology: OntologyDefinition,
    max_batch_count: int,
    max_batch_characters: int,
) -> ExtractedDocumentEvidence:
    """Run provider extraction outside SQL transactions and map all evidence."""

    if len(windows) > DOCUMENT_EXTRACTION_V1_MAX_CHUNKS:
        raise StructuralExtractionError("document extraction chunk cap exceeded")
    if (
        len(full_text) > DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS
        or sum(len(window.content) for window in windows)
        > DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS
    ):
        raise StructuralExtractionError("document extraction character cap exceeded")
    full_text = sanitize_graph_source_text(full_text)
    windows = tuple(
        replace(window, content=sanitize_graph_source_text(window.content))
        for window in windows
    )
    batches = batch_extraction_windows(
        windows,
        max_count=max_batch_count,
        max_characters=max_batch_characters,
    )
    mapped_entities: list[MappedEntityEvidence] = []
    mapped_relations: list[MappedRelationEvidence] = []
    diagnostic_counts: Counter[str] = Counter()
    raw_entity_count = 0
    raw_relation_count = 0

    for batch in batches:
        results = backend.extract_batch(
            tuple(window.content for window in batch),
            ontology=ontology,
        )
        if len(results) != len(batch):
            raise StructuralExtractionError(
                "provider returned a different number of window results"
            )
        for window, result in zip(batch, results, strict=True):
            raw_entity_count += len(result.entities)
            if raw_entity_count > DOCUMENT_EXTRACTION_V1_MAX_RAW_ENTITY_OBSERVATIONS:
                raise StructuralExtractionError(
                    "provider raw entity cap exceeded before candidate materialization"
                )
            raw_relation_count += len(result.relations)
            if (
                raw_relation_count
                > DOCUMENT_EXTRACTION_V1_MAX_RAW_RELATION_OBSERVATIONS
            ):
                raise StructuralExtractionError(
                    "provider raw relation cap exceeded before candidate "
                    "materialization"
                )
            fatal_codes = sorted(
                diagnostic.code
                for diagnostic in result.diagnostics
                if diagnostic.code in _FATAL_STRUCTURAL_DIAGNOSTICS
            )
            if fatal_codes:
                raise StructuralExtractionError(
                    "provider structural diagnostics: " + ", ".join(fatal_codes)
                )
            diagnostic_counts.update(
                diagnostic.code for diagnostic in result.diagnostics
            )
            local_entities = tuple(
                map_entity_candidate(window, candidate, full_text=full_text)
                for candidate in result.entities
            )
            mapped_entities.extend(local_entities)
            for relation in result.relations:
                head = _endpoint_entity(relation, "head", local_entities, ontology)
                tail = _endpoint_entity(relation, "tail", local_entities, ontology)
                if (
                    head is None
                    or tail is None
                    or head.identity_key == tail.identity_key
                ):
                    diagnostic_counts["unresolved_relation_endpoint"] += 1
                    continue
                mapped_relations.append(
                    MappedRelationEvidence(
                        document_id=window.document_id,
                        chunk_id=window.chunk_id,
                        relation_type=relation.relation_type,
                        head_identity=head.identity_key,
                        tail_identity=tail.identity_key,
                        confidence=relation.confidence,
                        observations=(
                            RelationObservation(
                                chunk_id=window.chunk_id,
                                confidence=relation.confidence,
                                modality=window.modality,
                                head_local_start=relation.head_start,
                                head_local_end=relation.head_end,
                                tail_local_start=relation.tail_start,
                                tail_local_end=relation.tail_end,
                            ),
                        ),
                    )
                )

    entities = deduplicate_mapped_entities(tuple(mapped_entities))
    if len(entities) > DOCUMENT_EXTRACTION_V1_MAX_ENTITIES:
        raise StructuralExtractionError("deduplicated entity cap exceeded")
    relations = _deduplicate_relations(mapped_relations)
    if len(relations) > DOCUMENT_EXTRACTION_V1_MAX_RELATIONS:
        raise StructuralExtractionError("deduplicated relation cap exceeded")
    return ExtractedDocumentEvidence(
        entities=entities,
        relations=relations,
        diagnostic_counts=dict(sorted(diagnostic_counts.items())),
        window_count=len(windows),
        batch_count=len(batches),
    )


def validate_build_identity(
    artifact,
    run,
    *,
    document_id,
    expected_source_hash: str,
    ontology_version: str,
) -> None:
    """Validate immutable request identity before any committed-result fast path."""

    from apps.knowledge_graph.models import GraphArtifact

    if run.artifact_id != artifact.pk:
        raise ValueError("build run must be owned by the destination artifact")
    if getattr(run, "build_key", None) != getattr(artifact, "build_key", None):
        raise ValueError("build run key does not match destination artifact")
    for field in ("build_generation", "orchestration_version"):
        if getattr(run, field, None) != getattr(artifact, field, None):
            raise ValueError(f"build run {field} does not match destination artifact")
    if artifact.scope_type != GraphArtifact.ScopeType.DOCUMENT:
        raise ValueError("mention extraction requires a document artifact")
    if str(artifact.scope_id) != str(document_id):
        raise ValueError("destination artifact document does not match")
    if artifact.source_hash != expected_source_hash:
        raise ValueError("destination artifact source hash does not match")
    if artifact.ontology_version != ontology_version:
        raise ValueError("destination artifact ontology does not match")
    if (
        run.assembly_version != artifact.assembly_version
        or run.assembly_config_checksum != artifact.assembly_config_checksum
    ):
        raise ValueError("build run assembly identity does not match destination")


def validate_build_lifecycle(
    artifact,
    run,
    *,
    lease_owner=None,
    lease_generation=None,
) -> None:
    """Validate mutable lifecycle state required to write raw evidence."""

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.builds import validate_build_lease

    validate_build_lease(run, lease_owner, lease_generation)

    if artifact.status != GraphArtifact.Status.BUILDING:
        raise ValueError("destination artifact must be building")
    if run.status != GraphBuildRun.Status.RUNNING:
        raise ValueError("destination build run must be running")
    if run.stage not in {
        GraphBuildRun.Stage.EXTRACTION,
        GraphBuildRun.Stage.EXTRACTING,
    }:
        raise ValueError("destination build run must be in extraction stage")


def _lock_extraction_orchestration_rows(
    artifact_id: int,
    build_run_id: int,
    document_id,
):
    """Lock D advisory, artifact, and run before any concrete source row."""

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.builds import (
        lock_document_graph_advisory_scope,
    )

    lock_document_graph_advisory_scope(document_id)
    artifact = GraphArtifact.objects.select_for_update().get(pk=artifact_id)
    run = GraphBuildRun.objects.select_for_update().get(pk=build_run_id)
    return artifact, run


def validate_build_destination(
    artifact,
    run,
    *,
    document_id,
    expected_source_hash: str,
    ontology_version: str,
) -> None:
    """Validate immutable identity and mutable state for a new evidence write."""

    validate_build_identity(
        artifact,
        run,
        document_id=document_id,
        expected_source_hash=expected_source_hash,
        ontology_version=ontology_version,
    )
    validate_build_lifecycle(artifact, run)


def extraction_commit_is_valid(
    run,
    *,
    entity_count: int,
    relation_count: int,
    evidence_fingerprint: str | None = None,
) -> bool:
    stats = run.stats if isinstance(run.stats, dict) else {}
    marker = stats.get("extraction_commit")
    ontology_checksum = getattr(run, "ontology_checksum", None)
    assembly_version = getattr(run, "assembly_version", None)
    assembly_config_checksum = getattr(run, "assembly_config_checksum", None)
    artifact_id = getattr(run, "artifact_id", None)
    artifact = getattr(run, "artifact", None) if artifact_id else None
    base_fields = {
        "version",
        "assembly_version",
        "assembly_config_checksum",
        "entity_mention_count",
        "relation_mention_count",
    }
    cap_fields = {
        "max_chunks",
        "max_characters",
        "max_entities",
        "max_relations",
    }
    scoped_v1 = bool(
        getattr(run, "orchestration_version", 0) == 1
        or getattr(artifact, "orchestration_version", 0) == 1
    )
    marker_fields = set(marker) if isinstance(marker, dict) else set()
    fields_valid = marker_fields == base_fields | cap_fields or (
        not scoped_v1 and marker_fields == base_fields
    )
    caps_valid = isinstance(marker, dict) and (
        marker_fields == base_fields
        or bool(
            marker.get("max_chunks") == DOCUMENT_EXTRACTION_V1_MAX_CHUNKS
            and marker.get("max_characters") == DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS
            and marker.get("max_entities") == DOCUMENT_EXTRACTION_V1_MAX_ENTITIES
            and marker.get("max_relations") == DOCUMENT_EXTRACTION_V1_MAX_RELATIONS
        )
    )
    marker_valid = (
        isinstance(marker, dict)
        and fields_valid
        and caps_valid
        and type(ontology_checksum) is str
        and len(ontology_checksum) == 64
        and all(character in "0123456789abcdef" for character in ontology_checksum)
        and (
            not artifact_id
            or (
                artifact is not None
                and ontology_checksum == getattr(artifact, "ontology_checksum", None)
                and assembly_version == getattr(artifact, "assembly_version", None)
                and assembly_config_checksum
                == getattr(artifact, "assembly_config_checksum", None)
            )
        )
        and type(marker.get("version")) is int
        and marker.get("version") == 1
        and type(marker.get("assembly_version")) is str
        and marker.get("assembly_version") == assembly_version
        and type(marker.get("assembly_config_checksum")) is str
        and len(marker.get("assembly_config_checksum")) == 64
        and all(
            character in "0123456789abcdef"
            for character in marker.get("assembly_config_checksum")
        )
        and marker.get("assembly_config_checksum") == assembly_config_checksum
        and type(marker.get("entity_mention_count")) is int
        and marker.get("entity_mention_count") == entity_count
        and type(marker.get("relation_mention_count")) is int
        and marker.get("relation_mention_count") == relation_count
    )
    if not marker_valid:
        return False
    if evidence_fingerprint is None:
        return True
    return bool(
        type(evidence_fingerprint) is str
        and len(evidence_fingerprint) == 64
        and all(character in "0123456789abcdef" for character in evidence_fingerprint)
        and stats.get("extraction_evidence_fingerprint") == evidence_fingerprint
    )


def _bounded_evidence_rows(values, maximum: int, label: str) -> tuple[object, ...]:
    """Count querysets before iterating and cap arbitrary iterables at cap + 1."""

    if type(maximum) is not int or maximum < 1:
        raise ValueError("evidence row cap must be a positive integer")
    query_count = getattr(values, "count", None)
    query_iterator = getattr(values, "iterator", None)
    if callable(query_count) and callable(query_iterator):
        count = query_count()
        if type(count) is not int or count < 0:
            raise ValueError(f"{label} count is invalid")
        if count > maximum:
            raise ValueError(f"{label} cap exceeded ({maximum})")
        records = tuple(
            islice(
                query_iterator(chunk_size=min(maximum, _QUERY_ITERATOR_BATCH_SIZE)),
                maximum + 1,
            )
        )
        if len(records) > maximum:
            raise ValueError(f"{label} cap exceeded ({maximum})")
        return records
    rows = tuple(islice(iter(values), maximum + 1))
    if len(rows) > maximum:
        raise ValueError(f"{label} cap exceeded ({maximum})")
    return rows


def extraction_evidence_fingerprint(entities, relations) -> str:
    """Hash every immutable persisted Task 7 evidence field in PK order."""

    entity_rows = tuple(
        sorted(
            _bounded_evidence_rows(
                entities,
                DOCUMENT_EXTRACTION_V1_MAX_ENTITIES,
                "extraction entity",
            ),
            key=lambda row: row.pk,
        )
    )
    relation_rows = tuple(
        sorted(
            _bounded_evidence_rows(
                relations,
                DOCUMENT_EXTRACTION_V1_MAX_RELATIONS,
                "extraction relation",
            ),
            key=lambda row: row.pk,
        )
    )
    payload = {
        "entities": [
            {
                "id": row.pk,
                "artifact_id": row.artifact_id,
                "document_id": str(row.document_id),
                "chunk_id": row.chunk_id,
                "start": row.start,
                "end": row.end,
                "position_basis": row.position_basis,
                "raw_text": row.raw_text,
                "normalized_text": row.normalized_text,
                "entity_type": row.entity_type,
                "extraction_confidence": row.extraction_confidence,
                "content_object_type_id": row.content_object_type_id,
                "content_object_id": (
                    str(row.content_object_id) if row.content_object_id else None
                ),
                "metadata": row.metadata,
            }
            for row in entity_rows
        ],
        "relations": [
            {
                "id": row.pk,
                "artifact_id": row.artifact_id,
                "document_id": str(row.document_id),
                "chunk_id": row.chunk_id,
                "head_id": row.head_id,
                "tail_id": row.tail_id,
                "relation_type": row.relation_type,
                "extraction_confidence": row.extraction_confidence,
                "metadata": row.metadata,
            }
            for row in relation_rows
        ],
    }
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _find_committed_extraction_run(artifact, *, for_update: bool = False):
    from apps.knowledge_graph.models import (
        EntityMention,
        GraphArtifact,
        GraphBuildRun,
        RelationMention,
    )

    run_query = GraphBuildRun.objects.filter(artifact=artifact)
    candidate_ids = set(run_query.order_by("-pk").values_list("pk", flat=True)[:1])
    candidate_ids.update(
        run_query.filter(stats__has_key="extraction_commit")
        .order_by("-pk")
        .values_list("pk", flat=True)[:2]
    )
    runs = GraphBuildRun.objects.filter(pk__in=tuple(sorted(candidate_ids))).order_by(
        "-pk"
    )
    if for_update:
        runs = runs.select_for_update()
    entity_query = EntityMention.objects.filter(artifact=artifact).order_by("pk")
    relation_query = RelationMention.objects.filter(artifact=artifact).order_by("pk")
    entity_count = entity_query.count()
    relation_count = relation_query.count()
    evidence_fingerprint = None
    if (
        getattr(artifact, "orchestration_version", None)
        == GraphArtifact.OrchestrationVersion.SCOPED_V1
    ):
        evidence_fingerprint = extraction_evidence_fingerprint(
            entity_query,
            relation_query,
        )
    return next(
        (
            run
            for run in runs
            if extraction_commit_is_valid(
                run,
                entity_count=entity_count,
                relation_count=relation_count,
                evidence_fingerprint=evidence_fingerprint,
            )
        ),
        None,
    )


def _terminal_mutation_policy(
    *, target_run_id: int, committed_run_id: int | None
) -> tuple[bool, bool]:
    """Return whether terminal bookkeeping may mutate artifact and target run."""

    if committed_run_id is None:
        return True, True
    return False, committed_run_id != target_run_id


def _get_concrete_document(document_id, *, for_update: bool = False):
    from apps.documents.models.document_types import DESCENDED_FROM_DOCUMENT

    matches = []
    for model in DESCENDED_FROM_DOCUMENT:
        queryset = model.objects
        if for_update:
            queryset = queryset.select_for_update()
        matches.extend(queryset.filter(id=document_id).order_by("pk")[:2])
        if len(matches) > 1:
            break
    if len(matches) != 1:
        raise DocumentResolutionError(
            "document UUID must resolve to exactly one concrete subtype; "
            f"found {len(matches)}"
        )
    return matches[0]


def _validate_source(document, expected_source_hash: str) -> None:
    calculated_hash = document.hash_fn(document.full_text)
    if not (
        getattr(document, "ingestion_complete", None) is True
        and expected_source_hash
        and document.full_text_hash == expected_source_hash == calculated_hash
    ):
        raise StaleSourceError(
            "document ingestion or source hash does not match expected content"
        )


def _ordered_chunks(document_id, *, for_update: bool = False):
    from django.db.models import Sum
    from django.db.models.functions import Coalesce, Length

    from apps.documents.models import TextChunk

    queryset = (
        TextChunk.objects.filter(doc_id=document_id)
        .only(
            "pk",
            "doc_id",
            "chunk_number",
            "start_position",
            "end_position",
            "modality",
            "content",
        )
        .order_by("chunk_number", "pk")
    )
    if for_update:
        queryset = queryset.select_for_update()
    if queryset.count() > DOCUMENT_EXTRACTION_V1_MAX_CHUNKS:
        raise StaleSourceError("ordered document chunk cap exceeded")
    totals = queryset.aggregate(total_characters=Coalesce(Sum(Length("content")), 0))
    total_characters = totals.get("total_characters")
    if type(total_characters) is not int or total_characters < 0:
        raise StaleSourceError("ordered document chunk character count is invalid")
    if total_characters > DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS:
        raise StaleSourceError("ordered document chunk character cap exceeded")
    chunks = tuple(
        islice(
            queryset.iterator(
                chunk_size=min(
                    DOCUMENT_EXTRACTION_V1_MAX_CHUNKS,
                    _QUERY_ITERATOR_BATCH_SIZE,
                )
            ),
            DOCUMENT_EXTRACTION_V1_MAX_CHUNKS + 1,
        )
    )
    if len(chunks) > DOCUMENT_EXTRACTION_V1_MAX_CHUNKS:
        raise StaleSourceError("ordered document chunk cap exceeded")
    return chunks


def _chunk_snapshot(chunks) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            chunk.pk,
            chunk.doc_id,
            chunk.chunk_number,
            chunk.start_position,
            chunk.end_position,
            chunk.modality,
            chunk.content,
        )
        for chunk in chunks
    )


def _windows_for_document(document, chunks) -> tuple[ExtractionWindow, ...]:
    from django.contrib.contenttypes.models import ContentType

    from apps.documents.models import TextChunk

    if any(chunk.doc_id != document.id for chunk in chunks):
        raise DocumentResolutionError(
            "ordered chunk provenance must match exactly one document UUID"
        )
    has_image = any(chunk.modality == TextChunk.Modality.IMAGE for chunk in chunks)
    content_type = None
    if has_image:
        content_type = ContentType.objects.get_for_model(
            document, for_concrete_model=False
        )
        allowed = {
            ("apps_documents", "documentfigure"),
            ("apps_documents", "handwrittennotesdocument"),
            ("apps_documents", "imageuploaddocument"),
        }
        if (content_type.app_label, content_type.model) not in allowed:
            raise ValueError(
                "image extraction requires an exact persisted image-document subtype"
            )

    windows = []
    for chunk in chunks:
        image = chunk.modality == TextChunk.Modality.IMAGE
        windows.append(
            ExtractionWindow(
                chunk_id=chunk.pk,
                document_id=chunk.doc_id,
                content=chunk.content,
                start_position=chunk.start_position,
                modality=chunk.modality,
                content_object_type_id=content_type.pk if image else None,
                content_object_app_label=(content_type.app_label if image else None),
                content_object_model=content_type.model if image else None,
                content_object_id=document.id if image else None,
            )
        )
    return tuple(windows)


def _resolve_ontology_definition(ontology_version: str, *, for_update=False):
    from apps.knowledge_graph.models import OntologyVersion
    from apps.knowledge_graph.services.ontology import (
        OntologyValidationError,
        load_ontology_yaml,
    )

    queryset = OntologyVersion.objects.filter(
        kind=OntologyVersion.Kind.GRAPH,
        version=ontology_version,
    )
    if for_update:
        queryset = queryset.select_for_update()
    record = queryset.first()
    if record is None:
        raise OntologySnapshotChangedError(
            "extraction requires a persisted active graph ontology version"
        )
    if record.status != OntologyVersion.Status.ACTIVE:
        raise OntologySnapshotChangedError(
            "selected graph ontology is no longer active"
        )
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    raw_yaml = metadata.get("yaml")
    if not isinstance(raw_yaml, str):
        raise OntologySnapshotChangedError(
            "persisted ontology is missing validated YAML"
        )
    try:
        definition = load_ontology_yaml(raw_yaml)
    except OntologyValidationError as exc:
        raise OntologySnapshotChangedError(
            "persisted ontology YAML is no longer valid"
        ) from exc
    if definition.version != record.version or definition.checksum != record.checksum:
        raise OntologySnapshotChangedError(
            "persisted ontology version or checksum does not match its YAML"
        )
    return definition


def resolve_ontology_definition(ontology_version: str, *, for_update=False):
    """Public Task 11 seam for a locked, validated ontology snapshot."""

    return _resolve_ontology_definition(ontology_version, for_update=for_update)


def _build_backend(settings):
    # The factory and provider implementation stay outside import-time web paths.
    from lib.knowledge_graph.extractors.factory import get_extraction_backend

    return get_extraction_backend(settings=settings)


def _extractor_identity(settings) -> str:
    identity = f"{settings.provider}:{settings.model_id}@{settings.model_revision}"
    if len(identity) > 128:
        raise ValueError("configured extractor identity exceeds artifact storage")
    return identity


def document_artifact_identity_values(
    document_id,
    expected_source_hash: str,
    ontology_version: str,
    ontology_checksum: str,
    *,
    settings,
) -> dict[str, object]:
    from apps.knowledge_graph.models import (
        ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM,
        ASSEMBLY_NOT_APPLICABLE_VERSION,
        GraphArtifact,
    )
    from apps.knowledge_graph.models.artifacts import graph_identity_checksum

    return {
        "scope_type": GraphArtifact.ScopeType.DOCUMENT,
        "scope_id": document_id,
        "source_hash": expected_source_hash,
        "ontology_version": ontology_version,
        "extractor_version": _extractor_identity(settings),
        "resolver_version": DOCUMENT_RESOLVER_VERSION,
        "filter_policy_version": "pending-v1",
        "ontology_checksum": ontology_checksum,
        "filter_policy_checksum": graph_identity_checksum(
            "document-filter-policy", "pending-v1"
        ),
        "resolution_config_checksum": graph_identity_checksum(
            "document-resolver", DOCUMENT_RESOLVER_VERSION
        ),
        "assembly_version": ASSEMBLY_NOT_APPLICABLE_VERSION,
        "assembly_config_checksum": ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM,
    }


# Backward-compatible internal seam retained for Task 7 callers/tests.
_artifact_identity_values = document_artifact_identity_values


def _create_build_destination(
    document_id,
    expected_source_hash: str,
    ontology_version: str,
    ontology_checksum: str,
    *,
    settings,
):
    from django.db import transaction
    from django.utils import timezone

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.builds import _lock_document_scope

    identity = _artifact_identity_values(
        document_id,
        expected_source_hash,
        ontology_version,
        ontology_checksum,
        settings=settings,
    )
    with transaction.atomic():
        _lock_document_scope(document_id)
        existing = (
            GraphArtifact.objects.select_for_update()
            .filter(**identity)
            .order_by("-build_generation", "-pk")
            .first()
        )
        if existing is not None:
            committed_run = _find_committed_extraction_run(
                existing,
                for_update=True,
            )
            if committed_run is not None:
                return existing, committed_run
            raise ExtractionInProgressError(
                "legacy extraction identity is already in progress"
            )
        latest_generation = (
            GraphArtifact.objects.filter(
                scope_type=GraphArtifact.ScopeType.DOCUMENT,
                scope_id=str(document_id),
            )
            .order_by("-build_generation", "-pk")
            .values_list("build_generation", flat=True)
            .first()
        )
        build_generation = (latest_generation or 0) + 1
        artifact = GraphArtifact.objects.create(
            status=GraphArtifact.Status.BUILDING,
            build_generation=build_generation,
            metadata={"stage": "raw_extraction"},
            **identity,
        )
        run = GraphBuildRun.objects.create(
            artifact=artifact,
            stage=GraphBuildRun.Stage.EXTRACTION,
            status=GraphBuildRun.Status.RUNNING,
            started_at=timezone.now(),
        )
    return artifact, run


def _relation_observation_metadata(relation: MappedRelationEvidence):
    return [
        {
            "chunk_id": observation.chunk_id,
            "confidence": observation.confidence,
            "modality": observation.modality,
            "head_local_start": observation.head_local_start,
            "head_local_end": observation.head_local_end,
            "tail_local_start": observation.tail_local_start,
            "tail_local_end": observation.tail_local_end,
        }
        for observation in relation.observations
    ]


def _persist_evidence(
    *,
    artifact,
    document,
    chunks,
    evidence: ExtractedDocumentEvidence,
):
    from apps.knowledge_graph.models import EntityMention, RelationMention

    chunks_by_id = {chunk.pk: chunk for chunk in chunks}
    mention_rows = [
        EntityMention(
            artifact=artifact,
            document_id=document.id,
            chunk=chunks_by_id[entity.chunk_id],
            start=entity.start,
            end=entity.end,
            position_basis=entity.position_basis,
            raw_text=entity.raw_text,
            normalized_text=entity.normalized_text,
            entity_type=entity.entity_type,
            extraction_confidence=entity.confidence,
            content_object_type_id=entity.content_object_type_id,
            content_object_id=entity.content_object_id,
            metadata={"observations": serialize_entity_observations(entity)},
        )
        for entity in evidence.entities
    ]
    EntityMention.objects.bulk_create(mention_rows)
    mentions_by_identity = {
        entity.identity_key: mention
        for entity, mention in zip(evidence.entities, mention_rows, strict=True)
    }
    relation_rows = [
        RelationMention(
            artifact=artifact,
            document_id=document.id,
            chunk=chunks_by_id[relation.chunk_id],
            head=mentions_by_identity[relation.head_identity],
            tail=mentions_by_identity[relation.tail_identity],
            relation_type=relation.relation_type,
            extraction_confidence=relation.confidence,
            metadata={"observations": _relation_observation_metadata(relation)},
        )
        for relation in evidence.relations
    ]
    RelationMention.objects.bulk_create(relation_rows)
    return (
        len(mention_rows),
        len(relation_rows),
        extraction_evidence_fingerprint(mention_rows, relation_rows),
    )


def _mark_terminal(
    artifact_id,
    build_run_id,
    *,
    artifact_status: str,
    run_status: str,
    error_code: str,
    lease_owner=None,
    lease_generation=None,
) -> None:
    from django.db import transaction
    from django.utils import timezone

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.builds import validate_build_lease

    with transaction.atomic():
        artifact = GraphArtifact.objects.select_for_update().get(pk=artifact_id)
        run = GraphBuildRun.objects.select_for_update().get(pk=build_run_id)
        if run.artifact_id != artifact.pk:
            return
        validate_build_lease(run, lease_owner, lease_generation)
        if (
            getattr(
                run,
                "orchestration_version",
                GraphArtifact.OrchestrationVersion.LEGACY,
            )
            == GraphArtifact.OrchestrationVersion.SCOPED_V1
        ):
            # The coordinator owns the typed scoped terminal transition.  A
            # stage primitive must never leave an extracting orchestration run
            # in a partially terminal status if the process dies between two
            # transactions.
            return
        committed_run = _find_committed_extraction_run(artifact, for_update=True)
        mutate_artifact, mutate_run = _terminal_mutation_policy(
            target_run_id=run.pk,
            committed_run_id=(committed_run.pk if committed_run is not None else None),
        )
        now = timezone.now()
        if mutate_artifact and artifact.status == GraphArtifact.Status.BUILDING:
            GraphArtifact.objects.filter(pk=artifact.pk).update(
                status=artifact_status,
                completed_at=now,
            )
        if mutate_run and run.status == GraphBuildRun.Status.RUNNING:
            GraphBuildRun.objects.filter(pk=run.pk).update(
                status=run_status,
                error_code=error_code,
                error_message=error_code,
                finished_at=now,
            )


def _safe_mark_terminal(
    artifact_id,
    build_run_id,
    *,
    artifact_status: str,
    run_status: str,
    error_code: str,
    lease_owner=None,
    lease_generation=None,
) -> bool:
    """Best-effort bookkeeping that cannot replace the triggering exception."""

    try:
        _mark_terminal(
            artifact_id,
            build_run_id,
            artifact_status=artifact_status,
            run_status=run_status,
            error_code=error_code,
            lease_owner=lease_owner,
            lease_generation=lease_generation,
        )
    except Exception:
        logger.exception(
            "obs.kg.terminal_update_failed",
            artifact_id=artifact_id,
            build_run_id=build_run_id,
            error_code=error_code,
        )
        return False
    return True


def extract_into_build(
    artifact_id,
    build_run_id,
    document_id,
    expected_source_hash,
    ontology_version,
    *,
    lease_owner=None,
    lease_generation=None,
):
    """Extract into one explicitly addressed building document artifact."""

    from django.db import transaction

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.builds import validate_build_lease
    from lib.knowledge_graph.config import load_extraction_settings

    artifact = GraphArtifact.objects.get(pk=artifact_id)
    run = GraphBuildRun.objects.get(pk=build_run_id)
    validate_build_identity(
        artifact,
        run,
        document_id=document_id,
        expected_source_hash=expected_source_hash,
        ontology_version=ontology_version,
    )
    validate_build_lease(run, lease_owner, lease_generation)
    committed_run = _find_committed_extraction_run(artifact)
    if committed_run is not None:
        return committed_run
    validate_build_lifecycle(
        artifact,
        run,
        lease_owner=lease_owner,
        lease_generation=lease_generation,
    )
    total_started = perf_counter()
    try:
        settings = load_extraction_settings()
        if artifact.extractor_version != _extractor_identity(settings):
            raise ValueError("destination artifact extractor does not match settings")
        if artifact.entity_mentions.exists() or artifact.relation_mentions.exists():
            raise ValueError("destination artifact already contains raw evidence")
        document = _get_concrete_document(document_id)
        _validate_source(document, expected_source_hash)
        ontology = _resolve_ontology_definition(ontology_version)
        if artifact.ontology_checksum != ontology.checksum:
            raise OntologySnapshotChangedError(
                "destination artifact ontology checksum does not match"
            )
        chunks = _ordered_chunks(document_id)
        initial_chunk_snapshot = _chunk_snapshot(chunks)
        windows = _windows_for_document(document, chunks)

        inference_started = perf_counter()
        backend = _build_backend(settings)
        evidence = collect_document_evidence(
            windows,
            full_text=document.full_text,
            backend=backend,
            ontology=ontology,
            max_batch_count=settings.batch_size,
            max_batch_characters=settings.max_batch_characters,
        )
        inference_seconds = perf_counter() - inference_started

        persistence_started = perf_counter()
        with transaction.atomic():
            locked_artifact, locked_run = _lock_extraction_orchestration_rows(
                artifact_id,
                build_run_id,
                document_id,
            )
            validate_build_identity(
                locked_artifact,
                locked_run,
                document_id=document_id,
                expected_source_hash=expected_source_hash,
                ontology_version=ontology_version,
            )
            validate_build_lease(locked_run, lease_owner, lease_generation)
            committed_run = _find_committed_extraction_run(
                locked_artifact, for_update=True
            )
            if committed_run is not None:
                return committed_run
            validate_build_lifecycle(
                locked_artifact,
                locked_run,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
            )
            if (
                locked_artifact.entity_mentions.exists()
                or locked_artifact.relation_mentions.exists()
            ):
                raise ValueError("destination artifact already contains raw evidence")
            locked_document = _get_concrete_document(document_id, for_update=True)
            try:
                _validate_source(locked_document, expected_source_hash)
            except StaleSourceError as exc:
                raise MidflightSourceChangedError(str(exc)) from exc
            locked_chunks = _ordered_chunks(document_id, for_update=True)
            if _chunk_snapshot(locked_chunks) != initial_chunk_snapshot:
                raise MidflightSourceChangedError(
                    "ordered document chunk snapshot changed during extraction"
                )
            locked_ontology = _resolve_ontology_definition(
                ontology_version, for_update=True
            )
            if locked_ontology.checksum != ontology.checksum:
                raise MidflightSourceChangedError(
                    "ontology snapshot changed during extraction"
                )
            if locked_artifact.ontology_checksum != locked_ontology.checksum:
                raise MidflightSourceChangedError(
                    "destination artifact ontology identity changed"
                )
            entity_count, relation_count, evidence_fingerprint = _persist_evidence(
                artifact=locked_artifact,
                document=locked_document,
                chunks=locked_chunks,
                evidence=evidence,
            )
            persistence_seconds = perf_counter() - persistence_started
            locked_run.stats = {
                "window_count": evidence.window_count,
                "batch_count": evidence.batch_count,
                "entity_mention_count": entity_count,
                "relation_mention_count": relation_count,
                "filtered_candidate_count": sum(evidence.diagnostic_counts.values()),
                "diagnostic_counts": evidence.diagnostic_counts,
                "provider": settings.provider,
                "model_id": settings.model_id,
                "model_revision": settings.model_revision,
                "ontology_checksum": ontology.checksum,
                "extraction_evidence_fingerprint": evidence_fingerprint,
                "extraction_commit": {
                    "version": 1,
                    "assembly_version": locked_artifact.assembly_version,
                    "assembly_config_checksum": (
                        locked_artifact.assembly_config_checksum
                    ),
                    "entity_mention_count": entity_count,
                    "relation_mention_count": relation_count,
                    "max_chunks": DOCUMENT_EXTRACTION_V1_MAX_CHUNKS,
                    "max_characters": DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS,
                    "max_entities": DOCUMENT_EXTRACTION_V1_MAX_ENTITIES,
                    "max_relations": DOCUMENT_EXTRACTION_V1_MAX_RELATIONS,
                },
            }
            locked_run.timings = {
                "inference_seconds": inference_seconds,
                "persistence_seconds": persistence_seconds,
                "total_seconds": perf_counter() - total_started,
            }
            locked_run.save(update_fields=["stats", "timings"])
        return GraphBuildRun.objects.get(pk=build_run_id)
    except StaleSourceError:
        _safe_mark_terminal(
            artifact_id,
            build_run_id,
            artifact_status=GraphArtifact.Status.STALE,
            run_status=GraphBuildRun.Status.CANCELLED,
            error_code="source_or_config_stale",
            lease_owner=lease_owner,
            lease_generation=lease_generation,
        )
        raise
    except Exception as exc:
        error_code = (
            "structural_extraction_failure"
            if isinstance(exc, StructuralExtractionError)
            else "provider_or_evidence_failure"
        )
        _safe_mark_terminal(
            artifact_id,
            build_run_id,
            artifact_status=GraphArtifact.Status.FAILED,
            run_status=GraphBuildRun.Status.FAILED,
            error_code=error_code,
            lease_owner=lease_owner,
            lease_generation=lease_generation,
        )
        raise


def extract_document_mentions(
    document_id,
    expected_source_hash,
    ontology_version,
):
    """Create a building destination and persist raw mention evidence into it."""

    from django.db import IntegrityError

    from apps.knowledge_graph.models import GraphArtifact
    from lib.knowledge_graph.config import load_extraction_settings

    document = _get_concrete_document(document_id)
    _validate_source(document, expected_source_hash)
    settings = load_extraction_settings()
    ontology = _resolve_ontology_definition(ontology_version)
    identity = _artifact_identity_values(
        document_id,
        expected_source_hash,
        ontology_version,
        ontology.checksum,
        settings=settings,
    )
    artifact = (
        GraphArtifact.objects.filter(**identity)
        .order_by("-build_generation", "-pk")
        .first()
    )
    if artifact is not None:
        committed_run = _find_committed_extraction_run(artifact)
        if committed_run is not None:
            return committed_run
        raise ExtractionInProgressError(
            "the immutable document extraction identity is already in progress"
        )
    try:
        artifact, run = _create_build_destination(
            document_id,
            expected_source_hash,
            ontology_version,
            ontology.checksum,
            settings=settings,
        )
    except IntegrityError:
        artifact = GraphArtifact.objects.get(**identity)
        committed_run = _find_committed_extraction_run(artifact)
        if committed_run is not None:
            return committed_run
        raise ExtractionInProgressError(
            "the immutable document extraction identity was created concurrently"
        ) from None
    return extract_into_build(
        artifact.pk,
        run.pk,
        document_id,
        expected_source_hash,
        ontology_version,
    )


__all__ = [
    "DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS",
    "DOCUMENT_EXTRACTION_V1_MAX_CHUNKS",
    "DOCUMENT_EXTRACTION_V1_MAX_ENTITIES",
    "DOCUMENT_EXTRACTION_V1_MAX_RAW_ENTITY_OBSERVATIONS",
    "DOCUMENT_EXTRACTION_V1_MAX_RAW_RELATION_OBSERVATIONS",
    "DOCUMENT_EXTRACTION_V1_MAX_RELATIONS",
    "ExtractedDocumentEvidence",
    "DocumentResolutionError",
    "ExtractionInProgressError",
    "MidflightSourceChangedError",
    "MappedRelationEvidence",
    "OntologySnapshotChangedError",
    "StaleSourceError",
    "StructuralExtractionError",
    "collect_document_evidence",
    "document_artifact_identity_values",
    "extraction_commit_is_valid",
    "extraction_evidence_fingerprint",
    "resolve_ontology_definition",
    "extract_document_mentions",
    "extract_into_build",
    "serialize_entity_observations",
    "validate_build_destination",
    "validate_build_identity",
    "validate_build_lifecycle",
]
