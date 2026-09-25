from __future__ import annotations

from dataclasses import replace

import pytest

from apps.knowledge_graph.tests.test_build_orchestration_postgres_races import (
    _collection_context,
    _collection_occurrence,
    _document_context,
    _document_occurrence,
    _patch_document_activation,
    _persist_document,
    database_required,
)

pytestmark = [pytest.mark.django_db(transaction=True), database_required]


def test_document_bootstrap_advances_past_a_detached_generation(monkeypatch):
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.graph.invalidation import (
        DocumentGraphCleanupResult,
        DocumentLifecycleRef,
        cleanup_document_graph_state,
    )
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services import builds

    collection, document, chunk = _persist_document(label="detached-doc-generation")
    initial_context = _document_context(document, chunk)
    _artifact, audit, _owner, _lease_generation = _document_occurrence(
        initial_context,
        generation=9,
        artifact_status=GraphArtifact.Status.BUILDING,
        run_stage=GraphBuildRun.Stage.QUEUED,
        run_status=GraphBuildRun.Status.PENDING,
    )

    cleanup = cleanup_document_graph_state(
        DocumentLifecycleRef(
            concrete_model_label=document._meta.label_lower,
            document_pkid=document.pkid,
            document_id=document.id,
        ),
        (collection.pk,),
        reason="test_generation_cleanup",
        expected_source_hash=document.full_text_hash,
    )
    assert isinstance(cleanup, DocumentGraphCleanupResult)
    audit.refresh_from_db()
    assert audit.artifact_id is None

    replacement_chunk = TextChunk.objects.create(
        content=document.full_text,
        start_position=0,
        end_position=len(document.full_text),
        chunk_number=0,
        modality=TextChunk.Modality.TEXT,
        doc_id=document.id,
        embedding=[0.0] * 1024,
    )
    next_context = _document_context(document, replacement_chunk)
    _patch_document_activation(monkeypatch, lambda: next_context)
    build_key = builds.derive_document_build_key(next_context.identity)

    artifact, run, owner, lease_generation, completed = (
        builds._bootstrap_document_build(next_context, build_key)
    )

    assert completed is False
    assert owner is not None and lease_generation is not None
    assert artifact.build_generation == run.build_generation == 10


def test_collection_bootstrap_advances_past_a_detached_generation():
    from apps.collections.models import Collection
    from apps.knowledge_graph.graph.invalidation import cleanup_collection_graph_state
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.models.inputs import collection_manifest_source_hash
    from apps.knowledge_graph.services import builds

    collection = Collection.objects.create(name="detached collection generation")
    base_context = _collection_context(collection)
    context = replace(
        base_context,
        identity=replace(
            base_context.identity,
            aggregate_source_signature=collection_manifest_source_hash(()),
        ),
    )
    _artifact, audit, _owner, _lease_generation = _collection_occurrence(
        context,
        generation=9,
        artifact_status=GraphArtifact.Status.BUILDING,
        run_stage=GraphBuildRun.Stage.QUEUED,
        run_status=GraphBuildRun.Status.PENDING,
    )

    cleanup_collection_graph_state(
        (collection.pk,),
        reason="test_generation_cleanup",
        all_artifacts=True,
    )
    audit.refresh_from_db()
    assert audit.artifact_id is None
    build_key = builds.derive_collection_build_key(context.identity)

    artifact, run, owner, lease_generation, completed = (
        builds._bootstrap_collection_build(context, build_key)
    )

    assert completed is False
    assert owner is not None and lease_generation is not None
    assert artifact.build_generation == run.build_generation == 10
