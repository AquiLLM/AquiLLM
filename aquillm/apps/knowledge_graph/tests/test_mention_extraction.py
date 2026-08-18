from __future__ import annotations

import inspect
import os
import socket
import uuid
from types import SimpleNamespace

import pytest
from django.conf import settings

from apps.knowledge_graph.extraction.pipeline import (
    StructuralExtractionError,
    collect_document_evidence,
    serialize_entity_observations,
)
from apps.knowledge_graph.extraction.windows import ExtractionWindow
from apps.knowledge_graph.services.ontology import (
    OntologyValidationError,
    load_ontology,
    load_ontology_yaml,
)
from lib.knowledge_graph.types import (
    EntityCandidate,
    ExtractionBatchResult,
    ExtractionDiagnostic,
    RelationCandidate,
)

ONTOLOGY_PATH = (
    __import__("pathlib").Path(__file__).parents[1] / "ontologies" / "research-v1.yaml"
)
DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")


def _ontology():
    return load_ontology(ONTOLOGY_PATH)


def _window(chunk_id: int, content: str, start: int) -> ExtractionWindow:
    return ExtractionWindow(
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        content=content,
        start_position=start,
        modality="text",
    )


def _result(
    *,
    model_start: int,
    dataset_start: int,
    confidence: float,
    diagnostics: tuple[ExtractionDiagnostic, ...] = (),
) -> ExtractionBatchResult:
    return ExtractionBatchResult(
        entities=(
            EntityCandidate(
                entity_type="model",
                text="Orion",
                start=model_start,
                end=model_start + 5,
                confidence=confidence,
            ),
            EntityCandidate(
                entity_type="dataset",
                text="MMLU",
                start=dataset_start,
                end=dataset_start + 4,
                confidence=confidence - 0.01,
            ),
        ),
        relations=(
            RelationCandidate(
                relation_type="uses_dataset",
                head_text="Orion",
                tail_text="MMLU",
                head_start=model_start,
                head_end=model_start + 5,
                tail_start=dataset_start,
                tail_end=dataset_start + 4,
                confidence=confidence - 0.02,
            ),
        ),
        diagnostics=diagnostics,
    )


class _Backend:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def extract_batch(self, texts, *, ontology):
        self.calls.append((texts, ontology.version))
        return tuple(next(self.results) for _text in texts)


def test_persisted_ontology_yaml_is_revalidated_without_a_temp_file():
    raw_yaml = ONTOLOGY_PATH.read_text(encoding="utf-8")

    definition = load_ontology_yaml(raw_yaml)

    assert definition.version == "1.0.0"
    assert definition.checksum == _ontology().checksum
    with pytest.raises(OntologyValidationError):
        load_ontology_yaml("version: 1.0.0\nversion: 2.0.0\n")


def test_collect_deduplicates_overlapping_mentions_and_relations_by_global_identity():
    full_text = "prefix Orion uses MMLU in evaluations."
    windows = (
        _window(10, "Orion uses MMLU", 7),
        _window(11, "prefix Orion uses MMLU", 0),
    )
    # The second window starts at zero but its provider-local evidence starts
    # seven code points later, mapping to the same global endpoint spans.
    backend = _Backend(
        (
            _result(model_start=0, dataset_start=11, confidence=0.82),
            _result(model_start=7, dataset_start=18, confidence=0.96),
        )
    )

    evidence = collect_document_evidence(
        windows,
        full_text=full_text,
        backend=backend,
        ontology=_ontology(),
        max_batch_count=1,
        max_batch_characters=100,
    )

    assert len(evidence.entities) == 2
    by_text = {entity.raw_text: entity for entity in evidence.entities}
    assert by_text["Orion"].chunk_id == 11
    assert [item.chunk_id for item in by_text["Orion"].observations] == [10, 11]
    assert len(evidence.relations) == 1
    relation = evidence.relations[0]
    assert relation.head_identity == by_text["Orion"].identity_key
    assert relation.tail_identity == by_text["MMLU"].identity_key
    assert relation.chunk_id == 11
    assert [item.chunk_id for item in relation.observations] == [10, 11]
    assert [call[0] for call in backend.calls] == [
        ("Orion uses MMLU",),
        ("prefix Orion uses MMLU",),
    ]


@pytest.mark.parametrize(
    "fatal_code", ["missing_entity_output", "missing_relation_output"]
)
def test_missing_structural_provider_sections_abort_the_whole_collection(fatal_code):
    diagnostic = ExtractionDiagnostic(
        code=fatal_code,
        candidate_kind="entity" if "entity" in fatal_code else "relation",
        input_index=0,
        details=(("reason", "provider_section_absent"),),
    )
    backend = _Backend(
        (
            _result(model_start=0, dataset_start=11, confidence=0.82),
            _result(
                model_start=0,
                dataset_start=11,
                confidence=0.83,
                diagnostics=(diagnostic,),
            ),
        )
    )

    with pytest.raises(StructuralExtractionError, match=fatal_code):
        collect_document_evidence(
            (
                _window(10, "Orion uses MMLU", 0),
                _window(11, "Orion uses MMLU", 0),
            ),
            full_text="Orion uses MMLU in evaluations.",
            backend=backend,
            ontology=_ontology(),
            max_batch_count=1,
            max_batch_characters=100,
        )


def test_candidate_diagnostics_are_counted_without_copying_private_details():
    diagnostic = ExtractionDiagnostic(
        code="malformed_entity_span",
        candidate_kind="entity",
        input_index=0,
        details=(("surface", "private unreleased model name"),),
    )
    evidence = collect_document_evidence(
        (_window(10, "Orion uses MMLU", 0),),
        full_text="Orion uses MMLU",
        backend=_Backend(
            (
                _result(
                    model_start=0,
                    dataset_start=11,
                    confidence=0.82,
                    diagnostics=(diagnostic,),
                ),
            )
        ),
        ontology=_ontology(),
        max_batch_count=2,
        max_batch_characters=100,
    )

    assert evidence.diagnostic_counts == {"malformed_entity_span": 1}
    assert "private unreleased model name" not in repr(evidence.diagnostic_counts)


def test_overlap_observation_metadata_is_json_safe_and_carries_coordinate_basis():
    evidence = collect_document_evidence(
        (_window(10, "Orion uses MMLU", 0),),
        full_text="Orion uses MMLU",
        backend=_Backend((_result(model_start=0, dataset_start=11, confidence=0.82),)),
        ontology=_ontology(),
        max_batch_count=2,
        max_batch_characters=100,
    )

    metadata = serialize_entity_observations(evidence.entities[0])

    assert metadata == [
        {
            "chunk_id": 10,
            "confidence": 0.82,
            "modality": "text",
            "position_basis": "document_global",
            "start": 0,
            "end": 5,
            "local_start": 0,
            "local_end": 5,
            "content_object_type_id": None,
            "content_object_id": None,
        }
    ]
    assert all(
        isinstance(value, (str, int, float, bool, type(None)))
        for observation in metadata
        for value in observation.values()
    )


def test_batch_character_guard_is_positive_and_environment_configured():
    from lib.knowledge_graph.config import (
        get_extractor_max_batch_characters,
        load_extraction_settings,
    )

    settings_from_env = load_extraction_settings(
        {"KG_GLINER2_MAX_BATCH_CHARACTERS": "12345"}
    )
    settings_from_invalid = load_extraction_settings(
        {"KG_GLINER2_MAX_BATCH_CHARACTERS": "0"}
    )

    assert settings_from_env.max_batch_characters == 12_345
    assert (
        get_extractor_max_batch_characters({"KG_GLINER2_MAX_BATCH_CHARACTERS": "23456"})
        == 23_456
    )
    assert settings_from_invalid.max_batch_characters > 0


def test_public_wrapper_and_explicit_destination_core_have_narrow_signatures():
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.extraction.pipeline import (
        extract_document_mentions,
        extract_into_build,
    )

    assert "extraction_commit_is_valid" in pipeline.__all__
    assert tuple(inspect.signature(extract_document_mentions).parameters) == (
        "document_id",
        "expected_source_hash",
        "ontology_version",
    )
    assert tuple(inspect.signature(extract_into_build).parameters) == (
        "artifact_id",
        "build_run_id",
        "document_id",
        "expected_source_hash",
        "ontology_version",
    )


def test_atomic_extraction_marker_rejects_partial_or_mismatched_evidence():
    from apps.knowledge_graph.extraction.pipeline import (
        extraction_commit_is_valid,
    )

    committed = SimpleNamespace(
        artifact_id=None,
        ontology_checksum="a" * 64,
        assembly_version="not-applicable",
        assembly_config_checksum="b" * 64,
        stats={
            "extraction_commit": {
                "version": 1,
                "assembly_version": "not-applicable",
                "assembly_config_checksum": "b" * 64,
                "entity_mention_count": 2,
                "relation_mention_count": 1,
            },
        },
    )
    partial = SimpleNamespace(stats={"entity_mention_count": 2})
    missing_checksum = SimpleNamespace(
        stats={"extraction_commit": committed.stats["extraction_commit"]}
    )

    class ChecksumSubclass(str):
        pass

    subclass_checksum = SimpleNamespace(
        artifact_id=None,
        ontology_checksum=ChecksumSubclass("a" * 64),
        assembly_version="not-applicable",
        assembly_config_checksum="b" * 64,
        stats=committed.stats,
    )

    assert extraction_commit_is_valid(committed, entity_count=2, relation_count=1)
    assert not extraction_commit_is_valid(committed, entity_count=1, relation_count=1)
    assert not extraction_commit_is_valid(partial, entity_count=2, relation_count=1)
    assert not extraction_commit_is_valid(
        missing_checksum, entity_count=2, relation_count=1
    )
    assert not extraction_commit_is_valid(
        subclass_checksum, entity_count=2, relation_count=1
    )


def test_interleaving_commit_preserves_completed_artifact_and_owning_run():
    from apps.knowledge_graph.extraction.pipeline import _terminal_mutation_policy

    assert _terminal_mutation_policy(target_run_id=7, committed_run_id=7) == (
        False,
        False,
    )
    assert _terminal_mutation_policy(target_run_id=8, committed_run_id=7) == (
        False,
        True,
    )
    assert _terminal_mutation_policy(target_run_id=8, committed_run_id=None) == (
        True,
        True,
    )


def test_sequential_duplicate_returns_committed_summary_without_creating_destination(
    monkeypatch,
):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import GraphArtifact
    from lib.knowledge_graph import config

    artifact = SimpleNamespace(pk=3)
    committed_run = SimpleNamespace(pk=7, stats={"extraction_commit": {"version": 1}})
    document = SimpleNamespace(
        id=DOCUMENT_ID,
        full_text="Orion",
        full_text_hash="a" * 64,
        hash_fn=lambda _text: "a" * 64,
    )

    class Query:
        def filter(self, **_kwargs):
            return self

        def first(self):
            return artifact

    monkeypatch.setattr(GraphArtifact, "objects", Query())
    monkeypatch.setattr(pipeline, "_get_concrete_document", lambda _id: document)
    monkeypatch.setattr(pipeline, "_validate_source", lambda *_args: None)
    monkeypatch.setattr(
        pipeline,
        "_resolve_ontology_definition",
        lambda *_args, **_kwargs: SimpleNamespace(checksum="b" * 64),
    )
    monkeypatch.setattr(config, "load_extraction_settings", lambda: object())
    monkeypatch.setattr(
        pipeline, "_artifact_identity_values", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        pipeline,
        "_find_committed_extraction_run",
        lambda _artifact: committed_run,
    )
    monkeypatch.setattr(
        pipeline,
        "_create_build_destination",
        lambda *_args, **_kwargs: pytest.fail(
            "duplicate must not create a destination"
        ),
    )

    assert (
        pipeline.extract_document_mentions(DOCUMENT_ID, "a" * 64, "1.0.0")
        is committed_run
    )


def test_interleaving_other_attempt_commit_only_terminalizes_the_losing_run(
    monkeypatch,
):
    from contextlib import nullcontext

    from django.db import transaction

    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    artifact = SimpleNamespace(
        pk=3,
        status=GraphArtifact.Status.BUILDING,
    )
    losing_run = SimpleNamespace(
        pk=8,
        artifact_id=3,
        status=GraphBuildRun.Status.RUNNING,
    )
    committed_run = SimpleNamespace(pk=7)
    artifact_updates = []
    run_updates = []

    class Manager:
        def __init__(self, value, updates):
            self.value = value
            self.updates = updates

        def select_for_update(self):
            return self

        def get(self, **_kwargs):
            return self.value

        def filter(self, **_kwargs):
            return self

        def update(self, **kwargs):
            self.updates.append(kwargs)

    monkeypatch.setattr(GraphArtifact, "objects", Manager(artifact, artifact_updates))
    monkeypatch.setattr(GraphBuildRun, "objects", Manager(losing_run, run_updates))
    monkeypatch.setattr(transaction, "atomic", nullcontext)
    monkeypatch.setattr(
        pipeline,
        "_find_committed_extraction_run",
        lambda _artifact, **_kwargs: committed_run,
    )

    pipeline._mark_terminal(
        artifact.pk,
        losing_run.pk,
        artifact_status=GraphArtifact.Status.FAILED,
        run_status=GraphBuildRun.Status.FAILED,
        error_code="provider_failure",
    )

    assert artifact_updates == []
    assert run_updates[0]["status"] == GraphBuildRun.Status.FAILED


def test_terminal_bookkeeping_failure_is_logged_without_masking_original(
    monkeypatch,
):
    from apps.knowledge_graph.extraction import pipeline

    monkeypatch.setattr(
        pipeline,
        "_mark_terminal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("database down")),
    )
    logged = []
    monkeypatch.setattr(
        pipeline.logger,
        "exception",
        lambda event, **kwargs: logged.append((event, kwargs)),
    )

    assert (
        pipeline._safe_mark_terminal(
            1,
            2,
            artifact_status="failed",
            run_status="failed",
            error_code="provider_failure",
        )
        is False
    )
    assert logged[0][0] == "obs.kg.terminal_update_failed"


@pytest.mark.parametrize(
    "status,checksum", [("superseded", None), ("active", "f" * 64)]
)
def test_ontology_deactivation_or_checksum_drift_is_classified_as_snapshot_stale(
    monkeypatch, status, checksum
):
    from apps.knowledge_graph.extraction.pipeline import (
        OntologySnapshotChangedError,
        _resolve_ontology_definition,
    )
    from apps.knowledge_graph.models import OntologyVersion

    definition = _ontology()
    record = SimpleNamespace(
        status=status,
        version=definition.version,
        checksum=checksum or definition.checksum,
        metadata={"yaml": definition.raw_yaml},
    )

    class Query:
        def filter(self, **_kwargs):
            return self

        def first(self):
            return record

    monkeypatch.setattr(OntologyVersion, "objects", Query())

    with pytest.raises(OntologySnapshotChangedError):
        _resolve_ontology_definition("1.0.0")


def test_malformed_ontology_metadata_is_classified_as_snapshot_stale(monkeypatch):
    from apps.knowledge_graph.extraction.pipeline import (
        OntologySnapshotChangedError,
        _resolve_ontology_definition,
    )
    from apps.knowledge_graph.models import OntologyVersion

    record = SimpleNamespace(
        status=OntologyVersion.Status.ACTIVE,
        version="1.0.0",
        checksum="f" * 64,
        metadata=None,
    )

    class Query:
        def filter(self, **_kwargs):
            return self

        def first(self):
            return record

    monkeypatch.setattr(OntologyVersion, "objects", Query())

    with pytest.raises(OntologySnapshotChangedError):
        _resolve_ontology_definition("1.0.0")


def test_source_freshness_requires_expected_stored_and_recomputed_hash_to_match():
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.extraction.pipeline import (
        StaleSourceError,
        _validate_source,
    )

    text = "Orion uses MMLU"
    source_hash = RawTextDocument.hash_fn(text)
    document = type(
        "SourceDocument",
        (),
        {
            "full_text": text,
            "full_text_hash": source_hash,
            "hash_fn": staticmethod(RawTextDocument.hash_fn),
        },
    )()

    _validate_source(document, source_hash)
    document.full_text_hash = "f" * 64
    with pytest.raises(StaleSourceError):
        _validate_source(document, source_hash)
    document.full_text_hash = source_hash
    document.full_text = "content changed without updating its stored hash"
    with pytest.raises(StaleSourceError):
        _validate_source(document, source_hash)


def test_concrete_document_resolution_rejects_two_rows_from_the_same_subtype(
    monkeypatch,
):
    from apps.documents.models import document_types
    from apps.knowledge_graph.extraction.pipeline import (
        DocumentResolutionError,
        _get_concrete_document,
    )

    class Query:
        def __init__(self, rows):
            self.rows = rows

        def filter(self, **_kwargs):
            return self

        def order_by(self, *_fields):
            return self

        def first(self):
            return self.rows[0] if self.rows else None

        def __getitem__(self, value):
            return self.rows[value]

    duplicate_type = type(
        "DuplicateDocumentType",
        (),
        {"objects": Query([object(), object()])},
    )
    empty_type = type("EmptyDocumentType", (), {"objects": Query([])})
    monkeypatch.setattr(
        document_types,
        "DESCENDED_FROM_DOCUMENT",
        [duplicate_type, empty_type],
    )

    with pytest.raises(DocumentResolutionError, match="exactly one"):
        _get_concrete_document(DOCUMENT_ID)


def test_ordered_chunk_query_loads_only_extraction_fields(monkeypatch):
    from apps.documents import models as document_models
    from apps.knowledge_graph.extraction.pipeline import _ordered_chunks

    calls = []

    class Query:
        def filter(self, **kwargs):
            calls.append(("filter", kwargs))
            return self

        def order_by(self, *fields):
            calls.append(("order_by", fields))
            return self

        def only(self, *fields):
            calls.append(("only", fields))
            return self

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(document_models.TextChunk, "objects", Query())

    assert _ordered_chunks(DOCUMENT_ID) == ()
    assert (
        "only",
        (
            "pk",
            "doc_id",
            "chunk_number",
            "start_position",
            "end_position",
            "modality",
            "content",
        ),
    ) in calls


def test_window_construction_rejects_chunks_from_a_different_document_uuid():
    from apps.knowledge_graph.extraction.pipeline import (
        DocumentResolutionError,
        _windows_for_document,
    )

    document = SimpleNamespace(id=DOCUMENT_ID)
    wrong_document_id = uuid.UUID("22222222-2222-4222-8222-222222222222")
    chunk = SimpleNamespace(
        pk=1,
        doc_id=wrong_document_id,
        content="Orion",
        start_position=0,
        modality="text",
    )

    with pytest.raises(DocumentResolutionError, match="chunk provenance"):
        _windows_for_document(document, (chunk,))


def _postgres_available() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _postgres_available() and not os.environ.get("KG_REQUIRE_POSTGRES_TESTS"),
    reason="configured PostgreSQL database is not reachable",
)


@pytest.mark.django_db(transaction=True)
@database_required
def test_duplicate_uuid_within_one_concrete_document_table_is_rejected(monkeypatch):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.extraction.pipeline import (
        DocumentResolutionError,
        _get_concrete_document,
    )

    monkeypatch.setattr(
        "apps.documents.tasks.chunking.create_chunks.delay",
        lambda *_args, **_kwargs: None,
    )
    user = User.objects.create_user(username=f"kg-duplicate-{uuid.uuid4()}")
    for index in range(2):
        text = f"duplicate UUID document {index}"
        document = RawTextDocument(
            id=DOCUMENT_ID,
            title=f"Duplicate {index}",
            full_text=text,
            full_text_hash=RawTextDocument.hash_fn(text),
            collection=Collection.objects.create(name=f"Duplicate KG {uuid.uuid4()}"),
            ingested_by=user,
            ingestion_complete=True,
        )
        document.save(dont_rechunk=True)

    with pytest.raises(DocumentResolutionError, match="exactly one"):
        _get_concrete_document(DOCUMENT_ID)


def _persist_active_ontology():
    from apps.knowledge_graph.models import OntologyVersion

    definition = _ontology()
    return OntologyVersion.objects.create(
        kind=OntologyVersion.Kind.GRAPH,
        version=definition.version,
        checksum=definition.checksum,
        status=OntologyVersion.Status.ACTIVE,
        metadata={"yaml": definition.raw_yaml},
    )


def _persist_text_document(monkeypatch, *, text="prefix Orion uses MMLU"):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument

    monkeypatch.setattr(
        "apps.documents.tasks.chunking.create_chunks.delay",
        lambda *_args, **_kwargs: None,
    )
    user = User.objects.create_user(username=f"kg-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"KG {uuid.uuid4()}")
    document = RawTextDocument(
        title="KG source",
        full_text=text,
        full_text_hash=RawTextDocument.hash_fn(text),
        collection=collection,
        ingested_by=user,
        ingestion_complete=True,
    )
    document.save(dont_rechunk=True)
    return document


def _persist_chunk(document, *, content, start, number, modality="text"):
    from apps.documents.models import TextChunk

    return TextChunk.objects.create(
        content=content,
        start_position=start,
        end_position=start + len(content),
        chunk_number=number,
        modality=modality,
        doc_id=document.id,
        embedding=[0.0] * 1024,
    )


@pytest.mark.django_db(transaction=True)
@database_required
def test_public_extraction_persists_deduped_mentions_exact_relation_endpoints_and_stats(
    monkeypatch,
):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import (
        EntityMention,
        GraphArtifact,
        RelationMention,
    )

    _persist_active_ontology()
    document = _persist_text_document(monkeypatch)
    _persist_chunk(document, content="prefix Orion uses MMLU", start=0, number=0)
    _persist_chunk(document, content="Orion uses MMLU", start=7, number=1)
    backend = _Backend(
        (
            _result(model_start=7, dataset_start=18, confidence=0.81),
            _result(model_start=0, dataset_start=11, confidence=0.96),
        )
    )
    monkeypatch.setattr(pipeline, "_build_backend", lambda _settings: backend)

    run = pipeline.extract_document_mentions(
        document.id,
        document.full_text_hash,
        "1.0.0",
    )

    run.refresh_from_db()
    run.artifact.refresh_from_db()
    assert run.status == run.Status.RUNNING
    assert run.artifact.status == GraphArtifact.Status.BUILDING
    assert run.stats["entity_mention_count"] == 2
    assert run.stats["relation_mention_count"] == 1
    assert run.stats["model_revision"]
    assert run.stats["ontology_checksum"] == _ontology().checksum
    assert run.stats["extraction_commit"] == {
        "version": 1,
        "assembly_version": run.artifact.assembly_version,
        "assembly_config_checksum": run.artifact.assembly_config_checksum,
        "entity_mention_count": 2,
        "relation_mention_count": 1,
    }
    mentions = list(EntityMention.objects.filter(artifact=run.artifact))
    assert len(mentions) == 2
    assert {len(mention.metadata["observations"]) for mention in mentions} == {2}
    relation = RelationMention.objects.get(artifact=run.artifact)
    assert relation.head_id in {mention.pk for mention in mentions}
    assert relation.tail_id in {mention.pk for mention in mentions}
    assert relation.head.raw_text == "Orion"
    assert relation.tail.raw_text == "MMLU"
    document.refresh_from_db()
    assert document.ingestion_complete is True

    repeated = pipeline.extract_document_mentions(
        document.id,
        document.full_text_hash,
        "1.0.0",
    )
    assert repeated.pk == run.pk
    assert GraphArtifact.objects.filter(scope_id=document.id).count() == 1
    assert EntityMention.objects.filter(artifact=run.artifact).count() == 2
    assert RelationMention.objects.filter(artifact=run.artifact).count() == 1


@pytest.mark.django_db(transaction=True)
@database_required
def test_structural_provider_failure_rolls_back_and_preserves_previous_active(
    monkeypatch,
):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import EntityMention, GraphArtifact, GraphBuildRun

    _persist_active_ontology()
    document = _persist_text_document(monkeypatch, text="Orion uses MMLU")
    _persist_chunk(document, content=document.full_text, start=0, number=0)
    previous = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document.id,
        status=GraphArtifact.Status.ACTIVE,
        source_hash="b" * 64,
        ontology_version="0.9.0",
        extractor_version="old",
        resolver_version="old",
        filter_policy_version="old",
    )
    diagnostic = ExtractionDiagnostic(
        code="missing_relation_output",
        candidate_kind="relation",
        input_index=0,
        details=(("reason", "provider_section_absent"),),
    )
    monkeypatch.setattr(
        pipeline,
        "_build_backend",
        lambda _settings: _Backend(
            (
                _result(
                    model_start=0,
                    dataset_start=11,
                    confidence=0.9,
                    diagnostics=(diagnostic,),
                ),
            )
        ),
    )

    with pytest.raises(StructuralExtractionError):
        pipeline.extract_document_mentions(
            document.id, document.full_text_hash, "1.0.0"
        )

    previous.refresh_from_db()
    failed_run = GraphBuildRun.objects.exclude(artifact=previous).get()
    failed_run.artifact.refresh_from_db()
    assert previous.status == GraphArtifact.Status.ACTIVE
    assert failed_run.status == GraphBuildRun.Status.FAILED
    assert failed_run.artifact.status == GraphArtifact.Status.FAILED
    assert not EntityMention.objects.filter(artifact=failed_run.artifact).exists()
    document.refresh_from_db()
    assert document.ingestion_complete is True


@pytest.mark.django_db(transaction=True)
@database_required
def test_initial_stale_hash_aborts_before_artifact_or_run_writes(monkeypatch):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    document = _persist_text_document(monkeypatch, text="Orion uses MMLU")
    before = (GraphArtifact.objects.count(), GraphBuildRun.objects.count())

    with pytest.raises(pipeline.StaleSourceError):
        pipeline.extract_document_mentions(document.id, "f" * 64, "1.0.0")

    assert (GraphArtifact.objects.count(), GraphBuildRun.objects.count()) == before


@pytest.mark.django_db(transaction=True)
@database_required
def test_midflight_source_change_marks_destination_stale_without_evidence(
    monkeypatch,
):
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import EntityMention, GraphArtifact, GraphBuildRun

    _persist_active_ontology()
    document = _persist_text_document(monkeypatch, text="Orion uses MMLU")
    _persist_chunk(document, content=document.full_text, start=0, number=0)

    class MutatingBackend:
        def extract_batch(self, texts, *, ontology):
            changed = "Changed while the provider was running"
            RawTextDocument.objects.filter(pk=document.pk).update(
                full_text=changed,
                full_text_hash=RawTextDocument.hash_fn(changed),
            )
            return (_result(model_start=0, dataset_start=11, confidence=0.9),)

    monkeypatch.setattr(pipeline, "_build_backend", lambda _settings: MutatingBackend())

    with pytest.raises(pipeline.MidflightSourceChangedError):
        pipeline.extract_document_mentions(
            document.id, document.full_text_hash, "1.0.0"
        )

    run = GraphBuildRun.objects.get()
    run.artifact.refresh_from_db()
    assert run.status == GraphBuildRun.Status.CANCELLED
    assert run.artifact.status == GraphArtifact.Status.STALE
    assert not EntityMention.objects.filter(artifact=run.artifact).exists()


@pytest.mark.django_db(transaction=True)
@database_required
def test_real_document_figure_image_chunk_persists_chunk_local_entity_and_relation(
    monkeypatch, settings, tmp_path
):
    from django.contrib.auth.models import User
    from django.core.files.uploadedfile import SimpleUploadedFile

    from apps.collections.models import Collection
    from apps.documents.models import DocumentFigure
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import EntityMention, RelationMention

    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        },
    }
    _persist_active_ontology()
    user = User.objects.create_user(username=f"kg-figure-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"KG Figure {uuid.uuid4()}")
    figure = DocumentFigure(
        title="Evaluation figure",
        full_text="OCR source text intentionally differs",
        full_text_hash=DocumentFigure.hash_fn("OCR source text intentionally differs"),
        collection=collection,
        ingested_by=user,
        ingestion_complete=True,
        image_file=SimpleUploadedFile("figure.png", b"fake-png"),
        source_format="pdf",
        figure_index=1,
    )
    figure.save(dont_rechunk=True)
    chunk = _persist_chunk(
        figure,
        content="Orion uses MMLU",
        start=99_000,
        number=0,
        modality="image",
    )
    monkeypatch.setattr(
        pipeline,
        "_build_backend",
        lambda _settings: _Backend(
            (_result(model_start=0, dataset_start=11, confidence=0.95),)
        ),
    )

    run = pipeline.extract_document_mentions(figure.id, figure.full_text_hash, "1.0.0")

    mentions = list(EntityMention.objects.filter(artifact=run.artifact))
    assert {(mention.start, mention.end) for mention in mentions} == {
        (0, 5),
        (11, 15),
    }
    assert {mention.position_basis for mention in mentions} == {"chunk_content"}
    assert {mention.content_object_id for mention in mentions} == {figure.id}
    assert {mention.content_object_type.model for mention in mentions} == {
        "documentfigure"
    }
    relation = RelationMention.objects.get(artifact=run.artifact)
    assert relation.chunk_id == chunk.pk
    assert relation.head_id in {mention.pk for mention in mentions}
    assert relation.tail_id in {mention.pk for mention in mentions}


def test_destination_validation_rejects_cross_artifact_build_run_before_sql():
    from apps.knowledge_graph.extraction.pipeline import validate_build_destination
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    artifact = GraphArtifact(
        pk=1,
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        status=GraphArtifact.Status.BUILDING,
        source_hash="a" * 64,
        ontology_version="1.0.0",
        extractor_version="extractor",
        resolver_version="pending",
        filter_policy_version="pending",
    )
    other_artifact = GraphArtifact(pk=2)
    run = GraphBuildRun(
        pk=3,
        artifact=other_artifact,
        status=GraphBuildRun.Status.RUNNING,
        stage=GraphBuildRun.Stage.EXTRACTION,
    )

    with pytest.raises(ValueError, match="owned by the destination artifact"):
        validate_build_destination(
            artifact,
            run,
            document_id=DOCUMENT_ID,
            expected_source_hash="a" * 64,
            ontology_version="1.0.0",
        )


@pytest.mark.parametrize(
    "request_overrides,cross_artifact_run,should_reject",
    [
        (
            {"document_id": uuid.UUID("22222222-2222-4222-8222-222222222222")},
            False,
            True,
        ),
        ({"expected_source_hash": "b" * 64}, False, True),
        ({"ontology_version": "2.0.0"}, False, True),
        ({}, True, True),
        ({}, False, False),
    ],
)
def test_committed_fast_path_validates_identity_before_bypassing_lifecycle(
    monkeypatch, request_overrides, cross_artifact_run, should_reject
):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    artifact = GraphArtifact(
        pk=1,
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        status=GraphArtifact.Status.ACTIVE,
        source_hash="a" * 64,
        ontology_version="1.0.0",
        extractor_version="extractor",
        resolver_version="pending",
        filter_policy_version="pending",
    )
    run_artifact = GraphArtifact(pk=2) if cross_artifact_run else artifact
    run = GraphBuildRun(
        pk=3,
        artifact=run_artifact,
        status=GraphBuildRun.Status.SUCCEEDED,
        stage=GraphBuildRun.Stage.EXTRACTION,
    )

    class Manager:
        def __init__(self, value):
            self.value = value

        def get(self, **_kwargs):
            return self.value

    monkeypatch.setattr(GraphArtifact, "objects", Manager(artifact))
    monkeypatch.setattr(GraphBuildRun, "objects", Manager(run))
    monkeypatch.setattr(
        pipeline,
        "_find_committed_extraction_run",
        lambda _artifact: run,
    )
    request = {
        "document_id": DOCUMENT_ID,
        "expected_source_hash": "a" * 64,
        "ontology_version": "1.0.0",
        **request_overrides,
    }

    if should_reject:
        with pytest.raises(ValueError):
            pipeline.extract_into_build(artifact.pk, run.pk, **request)
    else:
        assert pipeline.extract_into_build(artifact.pk, run.pk, **request) is run


def test_relation_model_accepts_an_endpoint_represented_by_an_overlapping_observation():
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import (
        EntityMention,
        GraphArtifact,
        RelationMention,
    )

    artifact = GraphArtifact(
        pk=1,
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        status=GraphArtifact.Status.BUILDING,
    )

    representative_chunk = TextChunk(pk=10, doc_id=DOCUMENT_ID, modality="text")
    relation_chunk = TextChunk(
        pk=11,
        doc_id=DOCUMENT_ID,
        modality="text",
        content="Orion uses MMLU",
        start_position=0,
    )
    observation = {
        "chunk_id": 11,
        "position_basis": "document_global",
        "start": 0,
        "end": 5,
        "local_start": 0,
        "local_end": 5,
        "modality": "text",
    }
    head = EntityMention(
        pk=20,
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=representative_chunk,
        start=0,
        end=5,
        position_basis="document_global",
        raw_text="Orion",
        metadata={"observations": [observation]},
    )
    tail = EntityMention(
        pk=21,
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=representative_chunk,
        start=11,
        end=15,
        position_basis="document_global",
        raw_text="MMLU",
        metadata={
            "observations": [
                {
                    **observation,
                    "start": 11,
                    "end": 15,
                    "local_start": 11,
                    "local_end": 15,
                }
            ]
        },
    )
    relation = RelationMention(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=relation_chunk,
        head=head,
        tail=tail,
        relation_type="uses_dataset",
        extraction_confidence=0.9,
    )

    relation.clean()


def test_relation_model_rejects_unrelated_or_image_endpoint_provenance():
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import (
        EntityMention,
        GraphArtifact,
        RelationMention,
    )

    artifact = GraphArtifact(
        pk=1,
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        status=GraphArtifact.Status.BUILDING,
    )

    representative_chunk = TextChunk(pk=10, doc_id=DOCUMENT_ID, modality="text")
    relation_chunk = TextChunk(pk=99, doc_id=DOCUMENT_ID, modality="text")
    endpoint = EntityMention(
        pk=20,
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=representative_chunk,
        start=0,
        end=5,
        position_basis="document_global",
        metadata={
            "observations": [
                {
                    "chunk_id": 99,
                    "position_basis": "chunk_content",
                    "start": 0,
                    "end": 5,
                    "modality": "image",
                }
            ]
        },
    )
    relation = RelationMention(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=relation_chunk,
        head=endpoint,
        tail=EntityMention(
            pk=21,
            artifact=artifact,
            document_id=DOCUMENT_ID,
            chunk=representative_chunk,
            start=11,
            end=15,
            position_basis="document_global",
            metadata={},
        ),
        relation_type="uses_dataset",
        extraction_confidence=0.9,
    )

    with pytest.raises(Exception, match="chunk|observation|provenance"):
        relation.clean()


def test_relation_model_rejects_spoofed_overlap_offsets_for_an_unrelated_chunk():
    from django.core.exceptions import ValidationError

    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import (
        EntityMention,
        GraphArtifact,
        RelationMention,
    )

    artifact = GraphArtifact(
        pk=1,
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        status=GraphArtifact.Status.BUILDING,
    )

    primary = TextChunk(pk=10, doc_id=DOCUMENT_ID, modality="text")
    unrelated = TextChunk(
        pk=11,
        doc_id=DOCUMENT_ID,
        modality="text",
        content="Elsewhere in this document",
        start_position=100,
    )
    metadata = {
        "observations": [
            {
                "chunk_id": 11,
                "position_basis": "document_global",
                "start": 0,
                "end": 5,
                "local_start": 0,
                "local_end": 5,
                "modality": "text",
            }
        ]
    }
    head = EntityMention(
        pk=20,
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=primary,
        start=0,
        end=5,
        position_basis="document_global",
        raw_text="Orion",
        metadata=metadata,
    )
    tail = EntityMention(
        pk=21,
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=primary,
        start=6,
        end=10,
        position_basis="document_global",
        raw_text="MMLU",
        metadata={
            "observations": [
                {
                    **metadata["observations"][0],
                    "start": 6,
                    "end": 10,
                    "local_start": 6,
                    "local_end": 10,
                }
            ]
        },
    )
    relation = RelationMention(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=unrelated,
        head=head,
        tail=tail,
        relation_type="uses_dataset",
        extraction_confidence=0.9,
    )

    with pytest.raises(ValidationError, match="observation|evidence"):
        relation.clean()
