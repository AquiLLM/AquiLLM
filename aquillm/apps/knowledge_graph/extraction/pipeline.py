"""Chunk-bounded document extraction and immutable mention persistence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING

from lib.knowledge_graph.types import RelationCandidate

from .windows import (
    ExtractionWindow,
    MappedEntityEvidence,
    batch_extraction_windows,
    deduplicate_mapped_entities,
    map_entity_candidate,
)

if TYPE_CHECKING:
    from lib.knowledge_graph.extractors.base import (
        ExtractionBackend,
        OntologyDefinition,
    )


_FATAL_STRUCTURAL_DIAGNOSTICS = frozenset(
    {"missing_entity_output", "missing_relation_output"}
)


class StructuralExtractionError(RuntimeError):
    """Raised when a provider omits a required per-window output section."""


class StaleSourceError(RuntimeError):
    """Raised before writes when the requested source snapshot is stale."""


class MidflightSourceChangedError(StaleSourceError):
    """Raised when document text or ordered chunks change during inference."""


class DocumentResolutionError(LookupError):
    """Raised when a UUID does not resolve to exactly one concrete document."""


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

    batches = batch_extraction_windows(
        windows,
        max_count=max_batch_count,
        max_characters=max_batch_characters,
    )
    mapped_entities: list[MappedEntityEvidence] = []
    mapped_relations: list[MappedRelationEvidence] = []
    diagnostic_counts: Counter[str] = Counter()

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

    return ExtractedDocumentEvidence(
        entities=deduplicate_mapped_entities(tuple(mapped_entities)),
        relations=_deduplicate_relations(mapped_relations),
        diagnostic_counts=dict(sorted(diagnostic_counts.items())),
        window_count=len(windows),
        batch_count=len(batches),
    )


def validate_build_destination(
    artifact,
    run,
    *,
    document_id,
    expected_source_hash: str,
    ontology_version: str,
) -> None:
    """Validate the explicitly addressed building artifact and owning run."""

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    if run.artifact_id != artifact.pk:
        raise ValueError("build run must be owned by the destination artifact")
    if artifact.scope_type != GraphArtifact.ScopeType.DOCUMENT:
        raise ValueError("mention extraction requires a document artifact")
    if artifact.scope_id != document_id:
        raise ValueError("destination artifact document does not match")
    if artifact.status != GraphArtifact.Status.BUILDING:
        raise ValueError("destination artifact must be building")
    if artifact.source_hash != expected_source_hash:
        raise ValueError("destination artifact source hash does not match")
    if artifact.ontology_version != ontology_version:
        raise ValueError("destination artifact ontology does not match")
    if run.status != GraphBuildRun.Status.RUNNING:
        raise ValueError("destination build run must be running")
    if run.stage != GraphBuildRun.Stage.EXTRACTION:
        raise ValueError("destination build run must be in extraction stage")


def _get_concrete_document(document_id, *, for_update: bool = False):
    from apps.documents.models.document_types import DESCENDED_FROM_DOCUMENT

    matches = []
    for model in DESCENDED_FROM_DOCUMENT:
        queryset = model.objects
        if for_update:
            queryset = queryset.select_for_update()
        document = queryset.filter(id=document_id).first()
        if document is not None:
            matches.append(document)
    if len(matches) != 1:
        raise DocumentResolutionError(
            "document UUID must resolve to exactly one concrete subtype; "
            f"found {len(matches)}"
        )
    return matches[0]


def _validate_source(document, expected_source_hash: str) -> None:
    calculated_hash = document.hash_fn(document.full_text)
    if not (
        expected_source_hash
        and document.full_text_hash == expected_source_hash == calculated_hash
    ):
        raise StaleSourceError("document source hash does not match expected content")


def _ordered_chunks(document_id, *, for_update: bool = False):
    from apps.documents.models import TextChunk

    queryset = TextChunk.objects.filter(doc_id=document_id).order_by(
        "chunk_number", "pk"
    )
    if for_update:
        queryset = queryset.select_for_update()
    return tuple(queryset)


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
        status=OntologyVersion.Status.ACTIVE,
    )
    if for_update:
        queryset = queryset.select_for_update()
    record = queryset.first()
    if record is None:
        raise OntologyValidationError(
            "extraction requires a persisted active graph ontology version"
        )
    raw_yaml = record.metadata.get("yaml")
    if not isinstance(raw_yaml, str):
        raise OntologyValidationError("persisted ontology is missing validated YAML")
    definition = load_ontology_yaml(raw_yaml)
    if definition.version != record.version or definition.checksum != record.checksum:
        raise OntologyValidationError(
            "persisted ontology version or checksum does not match its YAML"
        )
    return definition


def _build_backend(settings):
    # The factory and provider implementation stay outside import-time web paths.
    from lib.knowledge_graph.extractors.factory import get_extraction_backend

    return get_extraction_backend(settings=settings)


def _extractor_identity(settings) -> str:
    identity = f"{settings.provider}:{settings.model_id}@{settings.model_revision}"
    if len(identity) > 128:
        raise ValueError("configured extractor identity exceeds artifact storage")
    return identity


def _create_build_destination(
    document_id,
    expected_source_hash: str,
    ontology_version: str,
    *,
    settings,
):
    from django.db import transaction
    from django.utils import timezone

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    with transaction.atomic():
        artifact = GraphArtifact.objects.create(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id=document_id,
            status=GraphArtifact.Status.BUILDING,
            source_hash=expected_source_hash,
            ontology_version=ontology_version,
            extractor_version=_extractor_identity(settings),
            resolver_version="pending-v1",
            filter_policy_version="pending-v1",
            metadata={"stage": "raw_extraction"},
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
    return len(mention_rows), len(relation_rows)


def _mark_terminal(
    artifact_id,
    build_run_id,
    *,
    artifact_status: str,
    run_status: str,
    error_code: str,
) -> None:
    from django.db import transaction
    from django.utils import timezone

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    with transaction.atomic():
        artifact = GraphArtifact.objects.select_for_update().get(pk=artifact_id)
        run = GraphBuildRun.objects.select_for_update().get(pk=build_run_id)
        if run.artifact_id != artifact.pk:
            return
        now = timezone.now()
        if artifact.status == GraphArtifact.Status.BUILDING:
            GraphArtifact.objects.filter(pk=artifact.pk).update(
                status=artifact_status,
                completed_at=now,
            )
        if run.status == GraphBuildRun.Status.RUNNING:
            GraphBuildRun.objects.filter(pk=run.pk).update(
                status=run_status,
                error_code=error_code,
                error_message=error_code,
                finished_at=now,
            )


def extract_into_build(
    artifact_id,
    build_run_id,
    document_id,
    expected_source_hash,
    ontology_version,
):
    """Extract into one explicitly addressed building document artifact."""

    from django.db import transaction

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from lib.knowledge_graph.config import load_extraction_settings

    artifact = GraphArtifact.objects.get(pk=artifact_id)
    run = GraphBuildRun.objects.get(pk=build_run_id)
    validate_build_destination(
        artifact,
        run,
        document_id=document_id,
        expected_source_hash=expected_source_hash,
        ontology_version=ontology_version,
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
            locked_artifact = GraphArtifact.objects.select_for_update().get(
                pk=artifact_id
            )
            locked_run = GraphBuildRun.objects.select_for_update().get(pk=build_run_id)
            validate_build_destination(
                locked_artifact,
                locked_run,
                document_id=document_id,
                expected_source_hash=expected_source_hash,
                ontology_version=ontology_version,
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
            entity_count, relation_count = _persist_evidence(
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
            }
            locked_run.timings = {
                "inference_seconds": inference_seconds,
                "persistence_seconds": persistence_seconds,
                "total_seconds": perf_counter() - total_started,
            }
            locked_run.save(update_fields=["stats", "timings"])
        return GraphBuildRun.objects.get(pk=build_run_id)
    except StaleSourceError:
        _mark_terminal(
            artifact_id,
            build_run_id,
            artifact_status=GraphArtifact.Status.STALE,
            run_status=GraphBuildRun.Status.CANCELLED,
            error_code="source_stale",
        )
        raise
    except Exception as exc:
        error_code = (
            "structural_extraction_failure"
            if isinstance(exc, StructuralExtractionError)
            else "provider_or_evidence_failure"
        )
        _mark_terminal(
            artifact_id,
            build_run_id,
            artifact_status=GraphArtifact.Status.FAILED,
            run_status=GraphBuildRun.Status.FAILED,
            error_code=error_code,
        )
        raise


def extract_document_mentions(
    document_id,
    expected_source_hash,
    ontology_version,
):
    """Create a building destination and persist raw mention evidence into it."""

    from lib.knowledge_graph.config import load_extraction_settings

    document = _get_concrete_document(document_id)
    _validate_source(document, expected_source_hash)
    _resolve_ontology_definition(ontology_version)
    settings = load_extraction_settings()
    artifact, run = _create_build_destination(
        document_id,
        expected_source_hash,
        ontology_version,
        settings=settings,
    )
    return extract_into_build(
        artifact.pk,
        run.pk,
        document_id,
        expected_source_hash,
        ontology_version,
    )


__all__ = [
    "ExtractedDocumentEvidence",
    "DocumentResolutionError",
    "MidflightSourceChangedError",
    "MappedRelationEvidence",
    "StaleSourceError",
    "StructuralExtractionError",
    "collect_document_evidence",
    "extract_document_mentions",
    "extract_into_build",
    "serialize_entity_observations",
    "validate_build_destination",
]
