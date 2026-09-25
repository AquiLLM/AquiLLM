from __future__ import annotations

import uuid

import pytest
from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone

from apps.knowledge_graph.tests.test_build_orchestration_postgres_races import (
    _document_context,
    _persist_document,
    database_required,
)

pytestmark = [pytest.mark.django_db(transaction=True), database_required]


def _add_document(collection, *, label: str):
    from apps.documents.models import RawTextDocument, TextChunk

    user = User.objects.create_user(username=f"kg-eval-{label}-{uuid.uuid4().hex}")
    text = f"Atlas uses Northstar for {label}."
    document = RawTextDocument(
        title=f"KG eval {label}",
        full_text=text,
        full_text_hash=RawTextDocument.hash_fn(text),
        collection=collection,
        ingested_by=user,
        ingestion_complete=True,
    )
    document.save(dont_rechunk=True)
    chunk = TextChunk.objects.create(
        content=text,
        start_position=0,
        end_position=len(text),
        chunk_number=0,
        modality=TextChunk.Modality.TEXT,
        doc_id=document.id,
        embedding=[0.0] * 1024,
    )
    return document, chunk


def _request_snapshot(documents):
    return [
        {
            "document_id": str(document.id),
            "document_pkid": document.pkid,
            "model_label": document._meta.label_lower,
            "collection_id": document.collection_id,
            "source_hash": document.full_text_hash,
        }
        for document in sorted(documents, key=lambda row: row.id.int)
    ]


def _complete_eval_document(request, document, chunk):
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services import builds

    context = _document_context(document, chunk)
    build_key = builds.derive_document_build_key(context.identity)
    artifact = GraphArtifact.objects.create(
        **builds._document_artifact_values(context, build_key),
        build_generation=1,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        status=GraphArtifact.Status.BUILDING,
        rebuild_request=request,
        evaluation_only=True,
        metadata={
            "orchestration_version": 1,
            "ordered_chunk_signature": context.identity.ordered_chunk_signature,
            "ontology_activation_signature": (
                context.identity.ontology_activation_signature
            ),
        },
    )
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        rebuild_request=request,
        evaluation_only=True,
        stage=GraphBuildRun.Stage.VALIDATING,
        status=GraphBuildRun.Status.RUNNING,
        attempt=1,
        metadata={"orchestration_version": 1, "attempt_history": []},
        stage_marker={
            "orchestration_version": 1,
            "build_key": build_key,
            "stage_sequence": [GraphBuildRun.Stage.VALIDATING],
            "last_stage": GraphBuildRun.Stage.VALIDATING,
        },
    )
    builds._complete_evaluation_document_occurrence(artifact, run)
    artifact.refresh_from_db()
    return context, artifact


def _eval_manifest(*, document_count: int = 2):
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.graph.filtering import FilterPolicy
    from apps.knowledge_graph.models import GraphArtifact, GraphRebuildRequest
    from apps.knowledge_graph.models.inputs import (
        collection_input_source_signature,
        collection_manifest_source_hash,
        document_membership_signature,
    )
    from apps.knowledge_graph.resolution.collection import (
        COLLECTION_RESOLVER_VERSION,
        CollectionResolutionConfig,
        build_collection_snapshot,
    )

    collection, first, first_chunk = _persist_document(label="eval-manifest")
    document_rows = [(first, first_chunk)]
    for index in range(1, document_count):
        document_rows.append(_add_document(collection, label=f"doc-{index}"))
    documents = tuple(row[0] for row in document_rows)
    request = GraphRebuildRequest.objects.create(
        id=uuid.uuid4(),
        scope_type=GraphRebuildRequest.ScopeType.COLLECTION,
        scope_id=str(collection.pk),
        requested_documents=_request_snapshot(documents),
        document_count=len(documents),
        collection_count=1,
        status=GraphRebuildRequest.Status.RUNNING,
        evaluation_only=True,
        started_at=timezone.now(),
    )
    completed = tuple(
        _complete_eval_document(request, document, chunk)
        for document, chunk in document_rows
    )
    context = completed[0][0]
    sources = tuple(row[1] for row in completed)
    signatures = tuple(
        collection_input_source_signature(
            collection_id=collection.pk,
            document_id=document.id,
            document_artifact=source,
            membership_signature=document_membership_signature(document),
        )
        for document, source in zip(documents, sources, strict=True)
    )
    request.expected_aggregate_signature = collection_manifest_source_hash(signatures)
    request.save(update_fields=["expected_aggregate_signature", "updated_at"])
    filter_policy = FilterPolicy()
    resolution_config = CollectionResolutionConfig()
    assembly_config = assembly.AssemblyConfig()
    destination, _manifest = build_collection_snapshot(
        collection=collection,
        document_artifacts=sources,
        ontology=context.ontology,
        extractor_version=sources[0].extractor_version,
        resolver_version=COLLECTION_RESOLVER_VERSION,
        filter_policy=filter_policy,
        resolution_config=resolution_config,
        assembly_config=assembly_config,
        embedding_model_signature=(
            "test-local:model@rev:endpoint="
            f"{'e' * 64}:dims=1024:prep=kg-entity-v1:max_chars=8192:batch=64"
        ),
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        rebuild_request=request,
        evaluation_only=True,
    )
    return collection, documents, sources, destination, request, assembly_config


def _validate_manifest(collection, destination, config):
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.models import GraphArtifact

    with transaction.atomic():
        locked_collection = assembly.lock_collection_graph_scope(collection.pk)
        locked_destination = GraphArtifact.objects.select_for_update().get(
            pk=destination.pk
        )
        manifest = assembly._load_locked_manifest(locked_destination, config)
        return assembly._validate_locked_manifest(
            locked_collection,
            locked_destination,
            manifest,
            locked_destination.source_hash,
            config,
        )


def test_eval_manifest_validates_its_correlated_superseded_document_artifacts():
    collection, _documents, sources, destination, _request, config = _eval_manifest()

    validated = _validate_manifest(collection, destination, config)

    assert tuple(row.pk for row in validated) == tuple(row.pk for row in sources)


def test_eval_manifest_still_rejects_document_source_drift():
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.graph import assembly

    collection, documents, _sources, destination, _request, config = _eval_manifest()
    RawTextDocument.objects.filter(pkid=documents[0].pkid).update(
        full_text_hash="f" * 64
    )

    with pytest.raises(assembly.CollectionGraphSourceStaleError):
        _validate_manifest(collection, destination, config)


def test_eval_manifest_still_rejects_ordered_chunk_drift():
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.graph import assembly

    collection, documents, _sources, destination, _request, config = _eval_manifest()
    chunk = TextChunk.objects.get(doc_id=documents[0].id)
    TextChunk.objects.filter(pk=chunk.pk).update(
        content=f"{chunk.content} changed",
        end_position=chunk.end_position + len(" changed"),
    )

    with pytest.raises(assembly.CollectionGraphSourceStaleError):
        _validate_manifest(collection, destination, config)


def test_eval_manifest_still_rejects_collection_membership_drift():
    from apps.knowledge_graph.graph import assembly

    collection, _documents, _sources, destination, _request, config = _eval_manifest()
    _add_document(collection, label="late-member")

    with pytest.raises(assembly.CollectionGraphSourceStaleError):
        _validate_manifest(collection, destination, config)


def test_collection_failure_preserves_completed_eval_document_outcomes():
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    _collection, documents, _sources, _destination, request, _config = _eval_manifest()
    request.completed_document_count = len(documents)
    request.save(update_fields=["completed_document_count", "updated_at"])

    builds.record_rebuild_failure(request.pk, error_code="collection_rebuild_failed")

    request.refresh_from_db()
    assert request.status == GraphRebuildRequest.Status.PARTIAL
    assert request.error_code == "collection_rebuild_failed"
    assert request.completed_document_count == len(documents)
    assert request.terminal_failure_count == 0
    assert request.failed_collection_count == 1
    assert request.completed_at is not None
    terminal_state = (
        request.status,
        request.error_code,
        request.completed_document_count,
        request.terminal_failure_count,
        request.failed_collection_count,
        request.completed_at,
    )

    builds.record_rebuild_failure(request.pk, error_code="duplicate_delivery")
    request.refresh_from_db()

    assert (
        request.status,
        request.error_code,
        request.completed_document_count,
        request.terminal_failure_count,
        request.failed_collection_count,
        request.completed_at,
    ) == terminal_state


def test_collection_failure_uses_monotonic_completion_when_observation_is_lower(
    monkeypatch,
):
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    _collection, documents, _sources, _destination, request, _config = _eval_manifest(
        document_count=1
    )
    request.completed_document_count = len(documents)
    request.save(update_fields=["completed_document_count", "updated_at"])
    monkeypatch.setattr(
        builds, "_completed_request_document_artifacts", lambda *_args, **_kwargs: ()
    )

    builds.record_rebuild_failure(request.pk, error_code="collection_rebuild_failed")

    request.refresh_from_db()
    assert request.status == GraphRebuildRequest.Status.PARTIAL
    assert request.completed_document_count == len(documents)
    assert request.terminal_failure_count == 0
    assert request.failed_collection_count == 1
