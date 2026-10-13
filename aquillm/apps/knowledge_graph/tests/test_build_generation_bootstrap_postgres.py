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
        reason="document_chunks_replaced",
        expected_source_hash=document.full_text_hash,
    )
    assert isinstance(cleanup, DocumentGraphCleanupResult)
    audit.refresh_from_db()
    assert audit.artifact_id is None

    assert TextChunk.objects.get(pk=chunk.pk).content == document.full_text
    assert document.chunks.filter(pk=chunk.pk).exists()
    next_context = _document_context(document, chunk)
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


def test_committed_eval_source_drift_detaches_graph_and_retains_citation():
    """Ordinary committed edits clean up immediately, without a test transaction."""
    from apps.knowledge_graph.tests.test_eval_collection_failures_postgres import _eval_manifest
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    collection, documents, sources, destination, _request, _config = _eval_manifest()
    document = documents[0]
    chunk = TextChunk.objects.get(doc_id=document.id)
    old_text = chunk.content
    source_id = sources[0].pk
    run_id = GraphBuildRun.objects.get(artifact_id=source_id).pk
    document.full_text += " changed"
    document.save(dont_rechunk=True)
    assert not GraphArtifact.objects.filter(pk__in=(destination.pk, source_id)).exists()
    run = GraphBuildRun.objects.get(pk=run_id)
    assert run.artifact_id is None
    assert not GraphArtifact.objects.current().filter(scope_id=str(document.id)).exists()
    assert TextChunk.objects.get(pk=chunk.pk).content == old_text
    assert not document.chunks.exists()
