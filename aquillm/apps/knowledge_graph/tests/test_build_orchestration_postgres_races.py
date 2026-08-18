from __future__ import annotations

import os
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from threading import Barrier, Event, local
from types import SimpleNamespace

import pytest
from django.conf import settings
from django.db import close_old_connections, connection, transaction


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)

pytestmark = [pytest.mark.django_db(transaction=True), database_required]


def _embedding_signature() -> str:
    return (
        f"test-local:model@rev:endpoint={'e' * 64}:dims=1024:"
        "prep=kg-entity-v1:max_chars=8192:batch=64"
    )


@lru_cache(maxsize=1)
def _ontology():
    from apps.knowledge_graph.services.ontology import load_ontology

    path = Path(__file__).resolve().parents[1] / "ontologies" / "research-v1.yaml"
    return load_ontology(path)


def _persist_document(*, label: str = "race"):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk

    suffix = uuid.uuid4().hex
    user = User.objects.create_user(username=f"kg-{label}-{suffix}")
    collection = Collection.objects.create(name=f"KG {label} {suffix}")
    text = "Orion uses MMLU."
    document = RawTextDocument(
        title=f"KG {label}",
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
    return collection, document, chunk


def _document_context(document, chunk, *, activation_digit: str = "a"):
    from apps.knowledge_graph.models import (
        ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM,
        ASSEMBLY_NOT_APPLICABLE_VERSION,
        graph_identity_checksum,
    )
    from apps.knowledge_graph.resolution import DOCUMENT_RESOLVER_VERSION
    from apps.knowledge_graph.services import builds

    extractor_settings = SimpleNamespace(
        provider="test",
        model_id="test/gliner2",
        model_revision="revision-1",
        device="cpu",
        batch_size=8,
        max_batch_characters=8192,
        local_files_only=True,
    )
    ontology = _ontology()
    chunk_signature = builds.ordered_chunk_signature(
        (chunk,), concrete_model_label=document._meta.label_lower
    )
    identity = builds.DocumentBuildIdentity(
        document_id=document.id,
        source_hash=document.full_text_hash,
        ordered_chunk_signature=chunk_signature,
        extractor_package="gliner2==1.3.2",
        extractor_checkpoint=extractor_settings.model_id,
        extractor_model_revision=extractor_settings.model_revision,
        extractor_config_checksum=builds._identity_key(
            "kg-extractor-config-v1",
            {
                "provider": extractor_settings.provider,
                "device": extractor_settings.device,
                "batch_size": extractor_settings.batch_size,
                "max_batch_characters": (extractor_settings.max_batch_characters),
                "local_files_only": extractor_settings.local_files_only,
            },
        ),
        ontology_version=ontology.version,
        ontology_checksum=ontology.checksum,
        resolver_version=DOCUMENT_RESOLVER_VERSION,
        resolver_checksum=graph_identity_checksum(
            "document-resolver", DOCUMENT_RESOLVER_VERSION
        ),
        filter_version="pending-v1",
        filter_checksum=graph_identity_checksum("document-filter-policy", "pending-v1"),
        assembly_version=ASSEMBLY_NOT_APPLICABLE_VERSION,
        assembly_checksum=ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM,
        ontology_activation_signature=activation_digit * 64,
    )
    return builds._DocumentContext(
        identity=identity,
        collection_id=document.collection_id,
        ontology=ontology,
        settings=extractor_settings,
    )


def _document_occurrence(
    context,
    *,
    generation: int,
    artifact_status: str,
    run_stage: str,
    run_status: str,
    attempt: int = 1,
    claim: bool = False,
):
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services import builds

    build_key = builds.derive_document_build_key(context.identity)
    values = builds._document_artifact_values(context, build_key)
    artifact = GraphArtifact.objects.create(
        **values,
        build_generation=generation,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        status=artifact_status,
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
        stage=run_stage,
        status=run_status,
        attempt=attempt,
        metadata={"orchestration_version": 1, "attempt_history": []},
        stage_marker={
            "orchestration_version": 1,
            "build_key": build_key,
            "stage_sequence": [run_stage],
            "last_stage": run_stage,
        },
    )
    if not claim:
        return artifact, run, None, None
    owner = f"worker-{uuid.uuid4().hex}"
    with transaction.atomic():
        locked = GraphBuildRun.objects.select_for_update().get(pk=run.pk)
        owner, lease_generation = builds._claim_locked_run(locked, owner)
    run.refresh_from_db()
    return artifact, run, owner, lease_generation


def _collection_context(collection, *, source_digit: str = "c"):
    from apps.knowledge_graph.graph.assembly import (
        AssemblyConfig,
        assembly_config_checksum,
    )
    from apps.knowledge_graph.graph.filtering import (
        FilterPolicy,
        filter_policy_checksum,
    )
    from apps.knowledge_graph.resolution import COLLECTION_RESOLVER_VERSION
    from apps.knowledge_graph.resolution.collection import (
        CollectionResolutionConfig,
        resolution_config_checksum,
    )
    from apps.knowledge_graph.services import builds

    ontology = _ontology()
    filter_policy = FilterPolicy()
    resolution_config = CollectionResolutionConfig()
    assembly_config = AssemblyConfig()
    identity = builds.CollectionBuildIdentity(
        collection_id=collection.pk,
        aggregate_source_signature=source_digit * 64,
        extractor_version="empty-manifest-v1",
        ontology_version=ontology.version,
        ontology_checksum=ontology.checksum,
        resolver_version=COLLECTION_RESOLVER_VERSION,
        resolver_checksum=resolution_config_checksum(resolution_config),
        filter_version=filter_policy.version,
        filter_checksum=filter_policy_checksum(filter_policy),
        assembly_version=assembly_config.version,
        assembly_checksum=assembly_config_checksum(assembly_config),
        embedding_model_signature=_embedding_signature(),
        ontology_activation_signature="f" * 64,
    )
    return builds._CollectionContext(
        identity=identity,
        collection=collection,
        document_artifacts=(),
        ontology=ontology,
        filter_policy=filter_policy,
        resolution_config=resolution_config,
        assembly_config=assembly_config,
    )


def _collection_occurrence(
    context,
    *,
    generation: int = 1,
    artifact_status: str,
    run_stage: str,
    run_status: str,
    attempt: int = 1,
    claim: bool = False,
):
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services import builds

    identity = context.identity
    build_key = builds.derive_collection_build_key(identity)
    artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=identity.collection_id,
        build_key=build_key,
        build_generation=generation,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        status=artifact_status,
        source_hash=identity.aggregate_source_signature,
        ontology_version=identity.ontology_version,
        extractor_version=identity.extractor_version,
        resolver_version=identity.resolver_version,
        filter_policy_version=identity.filter_version,
        embedding_model_signature=identity.embedding_model_signature,
        ontology_checksum=identity.ontology_checksum,
        filter_policy_checksum=identity.filter_checksum,
        resolution_config_checksum=identity.resolver_checksum,
        assembly_version=identity.assembly_version,
        assembly_config_checksum=identity.assembly_checksum,
        metadata={
            "orchestration_version": 1,
            "ontology_activation_signature": (identity.ontology_activation_signature),
        },
    )
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        stage=run_stage,
        status=run_status,
        attempt=attempt,
        metadata={"orchestration_version": 1, "attempt_history": []},
        stage_marker={
            "orchestration_version": 1,
            "build_key": build_key,
            "stage_sequence": [run_stage],
            "last_stage": run_stage,
        },
    )
    if not claim:
        return artifact, run, None, None
    owner = f"worker-{uuid.uuid4().hex}"
    with transaction.atomic():
        locked = GraphBuildRun.objects.select_for_update().get(pk=run.pk)
        owner, lease_generation = builds._claim_locked_run(locked, owner)
    run.refresh_from_db()
    return artifact, run, owner, lease_generation


def _patch_document_activation(monkeypatch, context_for_thread, *, on_refresh=None):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.services import builds
    from lib.knowledge_graph import config as extraction_config

    monkeypatch.setattr(
        builds,
        "_document_context",
        lambda *_args, **_kwargs: context_for_thread(),
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_ontology_definition",
        lambda *_args, **_kwargs: context_for_thread().ontology,
    )
    monkeypatch.setattr(
        extraction_config,
        "load_extraction_settings",
        lambda: context_for_thread().settings,
    )
    monkeypatch.setattr(
        builds,
        "_document_commit_counts",
        lambda *_args: {
            "entity_mention_count": 0,
            "relation_mention_count": 0,
            "document_entity_count": 0,
            "membership_count": 0,
        },
    )
    if on_refresh is None:
        monkeypatch.setattr(
            builds, "_register_document_refresh_callbacks", lambda *_args: None
        )
    else:
        monkeypatch.setattr(
            builds,
            "_register_document_refresh_callbacks",
            lambda context, _run: transaction.on_commit(
                lambda: on_refresh(context.collection_id)
            ),
        )


def test_duplicate_collection_refreshes_share_one_live_occurrence(monkeypatch):
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.resolution import collection as resolution_collection
    from apps.knowledge_graph.services import builds

    collection = Collection.objects.create(name=f"duplicate {uuid.uuid4().hex}")
    context = _collection_context(collection)
    build_key = builds.derive_collection_build_key(context.identity)

    def fake_snapshot(**kwargs):
        artifact, _run, _owner, _generation = _collection_occurrence(
            context,
            generation=kwargs["build_generation"],
            artifact_status=GraphArtifact.Status.BUILDING,
            run_stage=GraphBuildRun.Stage.QUEUED,
            run_status=GraphBuildRun.Status.PENDING,
        )
        # The bootstrap owns run creation. Remove the helper-created run while
        # preserving the exact artifact identity returned by Task 9.
        GraphBuildRun.objects.filter(artifact=artifact).delete()
        return artifact, ()

    monkeypatch.setattr(
        resolution_collection, "build_collection_snapshot", fake_snapshot
    )
    barrier = Barrier(2)

    def bootstrap():
        close_old_connections()
        try:
            barrier.wait(timeout=10)
            try:
                artifact, run, _owner, _lease_generation, completed = (
                    builds._bootstrap_collection_build(context, build_key)
                )
                return "started", artifact.pk, run.pk, completed
            except builds.BuildInProgressError:
                return "in_progress", None, None, False
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(
            future.result(timeout=30)
            for future in (executor.submit(bootstrap), executor.submit(bootstrap))
        )

    assert sorted(item[0] for item in outcomes) == ["in_progress", "started"]
    assert (
        GraphArtifact.objects.filter(
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            scope_id=str(collection.pk),
            build_key=build_key,
        ).count()
        == 1
    )
    assert (
        GraphBuildRun.objects.filter(
            build_kind=GraphBuildRun.BuildKind.COLLECTION,
            scope_id=str(collection.pk),
            build_key=build_key,
        ).count()
        == 1
    )


def test_document_source_a_b_a_creates_a_new_occurrence_and_duplicate_joins_it(
    monkeypatch,
):
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services import builds

    _collection, document, chunk = _persist_document(label="document-a-b-a")
    context_a = _document_context(document, chunk)
    current = {"context": context_a}
    _patch_document_activation(monkeypatch, lambda: current["context"])

    def bootstrap_and_activate(context):
        current["context"] = context
        build_key = builds.derive_document_build_key(context.identity)
        artifact, run, owner, lease_generation, completed = (
            builds._bootstrap_document_build(context, build_key)
        )
        assert not completed
        assert owner is not None and lease_generation is not None
        for stage in (
            GraphBuildRun.Stage.EXTRACTING,
            GraphBuildRun.Stage.RESOLVING,
            GraphBuildRun.Stage.VALIDATING,
        ):
            run = builds._transition_run(
                run.pk,
                stage,
                lease_owner=owner,
                lease_generation=lease_generation,
            )
        activated, _counts = builds._activate_document_build(
            context,
            artifact.pk,
            run.pk,
            lease_owner=owner,
            lease_generation=lease_generation,
        )
        return activated, run

    first_a, first_a_run = bootstrap_and_activate(context_a)
    text_b = "Atlas uses MMLU."
    RawTextDocument.objects.filter(pk=document.pk).update(
        full_text=text_b,
        full_text_hash=RawTextDocument.hash_fn(text_b),
    )
    TextChunk.objects.filter(pk=chunk.pk).update(
        content=text_b,
        end_position=len(text_b),
    )
    document.refresh_from_db()
    chunk.refresh_from_db()
    context_b = _document_context(document, chunk)
    assert context_b.identity != context_a.identity
    second_b, second_b_run = bootstrap_and_activate(context_b)

    text_a = "Orion uses MMLU."
    RawTextDocument.objects.filter(pk=document.pk).update(
        full_text=text_a,
        full_text_hash=RawTextDocument.hash_fn(text_a),
    )
    TextChunk.objects.filter(pk=chunk.pk).update(
        content=text_a,
        end_position=len(text_a),
    )
    document.refresh_from_db()
    chunk.refresh_from_db()
    rollback_a = _document_context(document, chunk)
    assert rollback_a.identity == context_a.identity
    final_a, final_a_run = bootstrap_and_activate(rollback_a)

    current["context"] = rollback_a
    duplicate_artifact, duplicate_run, owner, lease_generation, completed = (
        builds._bootstrap_document_build(
            rollback_a,
            builds.derive_document_build_key(rollback_a.identity),
        )
    )
    artifacts = tuple(
        GraphArtifact.objects.filter(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id=str(document.id),
        ).order_by("build_generation")
    )
    runs = tuple(
        GraphBuildRun.objects.filter(
            build_kind=GraphBuildRun.BuildKind.DOCUMENT,
            scope_id=str(document.id),
        ).order_by("build_generation")
    )
    expected_a_key = builds.derive_document_build_key(context_a.identity)
    expected_b_key = builds.derive_document_build_key(context_b.identity)
    assert [row.build_generation for row in artifacts] == [1, 2, 3]
    assert [row.build_generation for row in runs] == [1, 2, 3]
    assert [row.build_key for row in artifacts] == [
        expected_a_key,
        expected_b_key,
        expected_a_key,
    ]
    assert [row.status for row in artifacts] == [
        GraphArtifact.Status.SUPERSEDED,
        GraphArtifact.Status.SUPERSEDED,
        GraphArtifact.Status.ACTIVE,
    ]
    assert (first_a.pk, second_b.pk, final_a.pk) == tuple(row.pk for row in artifacts)
    assert (first_a_run.pk, second_b_run.pk, final_a_run.pk) == tuple(
        row.pk for row in runs
    )
    assert completed is True
    assert owner is None and lease_generation is None
    assert duplicate_artifact.pk == final_a.pk
    assert duplicate_run.pk == final_a_run.pk
    assert tuple(
        GraphArtifact.objects.current()
        .filter(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id=str(document.id),
        )
        .values_list("pk", flat=True)
    ) == (final_a.pk,)
    assert len(artifacts) == len(runs) == 3


def test_collection_policy_a_b_a_creates_a_new_occurrence_and_duplicate_joins_it(
    monkeypatch,
):
    from apps.collections.models import Collection
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.graph.filtering import (
        FilterPolicy,
        filter_policy_checksum,
    )
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.models.inputs import collection_manifest_source_hash
    from apps.knowledge_graph.services import builds

    collection = Collection.objects.create(name=f"policy a-b-a {uuid.uuid4().hex}")
    base = _collection_context(collection)
    empty_manifest_source = collection_manifest_source_hash(())
    identity_a = replace(
        base.identity,
        aggregate_source_signature=empty_manifest_source,
    )
    context_a = replace(base, identity=identity_a)
    policy_b = FilterPolicy(utility_activation_threshold=0.35)
    identity_b = replace(
        identity_a,
        filter_checksum=filter_policy_checksum(policy_b),
    )
    context_b = replace(
        context_a,
        identity=identity_b,
        filter_policy=policy_b,
    )
    assert context_a.identity.aggregate_source_signature == (
        context_b.identity.aggregate_source_signature
    )
    assert builds.derive_collection_build_key(context_a.identity) != (
        builds.derive_collection_build_key(context_b.identity)
    )
    monkeypatch.setattr(
        assembly,
        "_validate_locked_complete_artifact",
        lambda **_kwargs: SimpleNamespace(),
    )

    def bootstrap_and_activate(context):
        build_key = builds.derive_collection_build_key(context.identity)
        artifact, run, owner, lease_generation, completed = (
            builds._bootstrap_collection_build(context, build_key)
        )
        assert not completed
        assert owner is not None and lease_generation is not None
        for stage in (
            GraphBuildRun.Stage.SNAPSHOTTING,
            GraphBuildRun.Stage.RESOLVING,
            GraphBuildRun.Stage.ASSEMBLING,
            GraphBuildRun.Stage.VALIDATING,
        ):
            run = builds._transition_run(
                run.pk,
                stage,
                lease_owner=owner,
                lease_generation=lease_generation,
            )
        assembly.activate_collection_graph(
            collection.pk,
            run.pk,
            context.identity.aggregate_source_signature,
            ontology=context.ontology,
            config=context.assembly_config,
            lease_owner=owner,
            lease_generation=lease_generation,
        )
        artifact.refresh_from_db()
        run.refresh_from_db()
        return artifact, run

    first_a, first_a_run = bootstrap_and_activate(context_a)
    second_b, second_b_run = bootstrap_and_activate(context_b)
    final_a, final_a_run = bootstrap_and_activate(context_a)
    duplicate_artifact, duplicate_run, owner, lease_generation, completed = (
        builds._bootstrap_collection_build(
            context_a,
            builds.derive_collection_build_key(context_a.identity),
        )
    )
    artifacts = tuple(
        GraphArtifact.objects.filter(
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            scope_id=str(collection.pk),
        ).order_by("build_generation")
    )
    runs = tuple(
        GraphBuildRun.objects.filter(
            build_kind=GraphBuildRun.BuildKind.COLLECTION,
            scope_id=str(collection.pk),
        ).order_by("build_generation")
    )
    expected_a_key = builds.derive_collection_build_key(context_a.identity)
    expected_b_key = builds.derive_collection_build_key(context_b.identity)
    assert [row.build_generation for row in artifacts] == [1, 2, 3]
    assert [row.build_generation for row in runs] == [1, 2, 3]
    assert [row.build_key for row in artifacts] == [
        expected_a_key,
        expected_b_key,
        expected_a_key,
    ]
    assert {row.source_hash for row in artifacts} == {empty_manifest_source}
    assert [row.status for row in artifacts] == [
        GraphArtifact.Status.SUPERSEDED,
        GraphArtifact.Status.SUPERSEDED,
        GraphArtifact.Status.ACTIVE,
    ]
    assert (first_a.pk, second_b.pk, final_a.pk) == tuple(row.pk for row in artifacts)
    assert (first_a_run.pk, second_b_run.pk, final_a_run.pk) == tuple(
        row.pk for row in runs
    )
    assert completed is True
    assert owner is None and lease_generation is None
    assert duplicate_artifact.pk == final_a.pk
    assert duplicate_run.pk == final_a_run.pk
    assert tuple(
        GraphArtifact.objects.current_collection(collection.pk).values_list(
            "pk", flat=True
        )
    ) == (final_a.pk,)
    assert len(artifacts) == len(runs) == 3


@pytest.mark.parametrize("completion_order", [("older", "newer"), ("newer", "older")])
def test_concurrent_document_activations_are_generation_fenced_in_both_orders(
    monkeypatch,
    completion_order,
):
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services import builds

    _collection, document, chunk = _persist_document(label="activation-order")
    base_context = _document_context(document, chunk, activation_digit="a")
    contexts = {
        "older": base_context,
        "newer": replace(
            base_context,
            identity=replace(
                base_context.identity, ontology_activation_signature="b" * 64
            ),
        ),
    }
    occurrences = {
        "older": _document_occurrence(
            contexts["older"],
            generation=1,
            artifact_status=GraphArtifact.Status.BUILDING,
            run_stage=GraphBuildRun.Stage.VALIDATING,
            run_status=GraphBuildRun.Status.RUNNING,
            claim=True,
        ),
        "newer": _document_occurrence(
            contexts["newer"],
            generation=2,
            artifact_status=GraphArtifact.Status.BUILDING,
            run_stage=GraphBuildRun.Stage.VALIDATING,
            run_status=GraphBuildRun.Status.RUNNING,
            claim=True,
        ),
    }
    thread_state = local()
    _patch_document_activation(monkeypatch, lambda: thread_state.context)
    start = Barrier(3)
    gates = {name: Event() for name in contexts}

    def activate(name):
        close_old_connections()
        thread_state.context = contexts[name]
        artifact, run, owner, lease_generation = occurrences[name]
        try:
            start.wait(timeout=10)
            assert gates[name].wait(timeout=20)
            try:
                builds._activate_document_build(
                    contexts[name],
                    artifact.pk,
                    run.pk,
                    lease_owner=owner,
                    lease_generation=lease_generation,
                )
                return "active"
            except builds.StaleBuildError:
                return "stale"
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {name: executor.submit(activate, name) for name in ("older", "newer")}
        start.wait(timeout=10)
        gates[completion_order[0]].set()
        first = futures[completion_order[0]].result(timeout=30)
        gates[completion_order[1]].set()
        second = futures[completion_order[1]].result(timeout=30)

    occurrences["older"][0].refresh_from_db()
    occurrences["newer"][0].refresh_from_db()
    occurrences["older"][1].refresh_from_db()
    occurrences["newer"][1].refresh_from_db()
    assert {first, second} == {"active", "stale"}
    assert occurrences["older"][0].status == GraphArtifact.Status.STALE
    assert occurrences["newer"][0].status == GraphArtifact.Status.ACTIVE
    assert occurrences["older"][1].stage == GraphBuildRun.Stage.STALE
    assert occurrences["newer"][1].stage == GraphBuildRun.Stage.ACTIVE


def test_collection_snapshot_racing_new_document_activation_is_never_current(
    monkeypatch,
):
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.resolution.collection import (
        CollectionResolutionPersistenceError,
        build_collection_snapshot,
    )
    from apps.knowledge_graph.services import builds

    collection, document, chunk = _persist_document(label="collection-v-document")
    old_context = _document_context(document, chunk, activation_digit="a")
    new_context = replace(
        old_context,
        identity=replace(old_context.identity, ontology_activation_signature="b" * 64),
    )
    old_artifact, _old_run, _owner, _generation = _document_occurrence(
        old_context,
        generation=1,
        artifact_status=GraphArtifact.Status.ACTIVE,
        run_stage=GraphBuildRun.Stage.ACTIVE,
        run_status=GraphBuildRun.Status.SUCCEEDED,
    )
    new_artifact, new_run, owner, lease_generation = _document_occurrence(
        new_context,
        generation=2,
        artifact_status=GraphArtifact.Status.BUILDING,
        run_stage=GraphBuildRun.Stage.VALIDATING,
        run_status=GraphBuildRun.Status.RUNNING,
        claim=True,
    )
    thread_state = local()
    refreshes = []
    _patch_document_activation(
        monkeypatch,
        lambda: getattr(thread_state, "context", new_context),
        on_refresh=refreshes.append,
    )
    context = _collection_context(collection)
    barrier = Barrier(2)

    def snapshot():
        close_old_connections()
        try:
            barrier.wait(timeout=10)
            try:
                artifact, manifest = build_collection_snapshot(
                    collection=collection,
                    document_artifacts=(old_artifact,),
                    ontology=context.ontology,
                    extractor_version=old_artifact.extractor_version,
                    resolver_version=context.identity.resolver_version,
                    filter_policy=context.filter_policy,
                    resolution_config=context.resolution_config,
                    assembly_config=context.assembly_config,
                    embedding_model_signature=(
                        context.identity.embedding_model_signature
                    ),
                    build_key=builds.derive_collection_build_key(context.identity),
                    orchestration_version=(
                        GraphArtifact.OrchestrationVersion.SCOPED_V1
                    ),
                )
                return "snapshotted", artifact.pk, tuple(row.pk for row in manifest)
            except CollectionResolutionPersistenceError:
                return "source_changed", None, ()
        finally:
            close_old_connections()

    def activate():
        close_old_connections()
        thread_state.context = new_context
        try:
            barrier.wait(timeout=10)
            builds._activate_document_build(
                new_context,
                new_artifact.pk,
                new_run.pk,
                lease_owner=owner,
                lease_generation=lease_generation,
            )
            return "activated"
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshot_future = executor.submit(snapshot)
        activation_future = executor.submit(activate)
        snapshot_outcome = snapshot_future.result(timeout=30)
        assert activation_future.result(timeout=30) == "activated"

    old_artifact.refresh_from_db()
    new_artifact.refresh_from_db()
    assert old_artifact.status == GraphArtifact.Status.SUPERSEDED
    assert new_artifact.status == GraphArtifact.Status.ACTIVE
    assert refreshes == [collection.pk]
    if snapshot_outcome[0] == "snapshotted":
        from apps.knowledge_graph.models import CollectionArtifactInput

        collection_artifact = GraphArtifact.objects.get(pk=snapshot_outcome[1])
        manifest = tuple(
            CollectionArtifactInput.objects.filter(pk__in=snapshot_outcome[2]).order_by(
                "pk"
            )
        )
        with (
            transaction.atomic(),
            pytest.raises(
                assembly.CollectionGraphSourceStaleError,
                match="active document artifact snapshot changed",
            ),
        ):
            assembly._validate_locked_manifest(
                collection,
                collection_artifact,
                manifest,
                collection_artifact.source_hash,
                context.assembly_config,
            )
    else:
        assert snapshot_outcome[0] == "source_changed"


def test_document_activation_detects_real_chunk_drift(monkeypatch):
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services import builds

    _collection, document, chunk = _persist_document(label="chunk-drift")
    context = _document_context(document, chunk)
    artifact, run, owner, lease_generation = _document_occurrence(
        context,
        generation=1,
        artifact_status=GraphArtifact.Status.BUILDING,
        run_stage=GraphBuildRun.Stage.VALIDATING,
        run_status=GraphBuildRun.Status.RUNNING,
        claim=True,
    )
    _patch_document_activation(monkeypatch, lambda: context)
    TextChunk.objects.filter(pk=chunk.pk).update(content="MMLU uses Orion.")

    with pytest.raises(builds.StaleBuildError, match="chunks changed"):
        builds._activate_document_build(
            context,
            artifact.pk,
            run.pk,
            lease_owner=owner,
            lease_generation=lease_generation,
        )

    artifact.refresh_from_db()
    run.refresh_from_db()
    assert artifact.status == GraphArtifact.Status.BUILDING
    assert run.stage == GraphBuildRun.Stage.VALIDATING


def test_post_resolution_resume_skips_providers_and_activates_same_document(
    monkeypatch,
):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.resolution import coreference, persistence
    from apps.knowledge_graph.services import builds

    _collection, document, chunk = _persist_document(label="document-resume")
    context = _document_context(document, chunk)
    artifact, run, owner, lease_generation = _document_occurrence(
        context,
        generation=1,
        artifact_status=GraphArtifact.Status.BUILDING,
        run_stage=GraphBuildRun.Stage.QUEUED,
        run_status=GraphBuildRun.Status.PENDING,
        attempt=2,
        claim=True,
    )
    run.stats = {"fixture_commit_state": "validated_by_injected_inspectors"}
    run.save(update_fields=["stats"])
    _patch_document_activation(monkeypatch, lambda: context)
    monkeypatch.setattr(builds, "_document_context", lambda *_args: context)
    monkeypatch.setattr(
        builds,
        "_bootstrap_document_build",
        lambda *_args: (artifact, run, owner, lease_generation, False),
    )
    monkeypatch.setattr(
        builds,
        "_document_extraction_commit_state",
        lambda *_args: builds.CommitMarkerState.VALID,
    )
    monkeypatch.setattr(
        builds,
        "_document_resolution_commit_state",
        lambda *_args: builds.CommitMarkerState.VALID,
    )

    def provider_must_not_run(*_args, **_kwargs):
        pytest.fail("a committed pre-activation resume called a provider")

    monkeypatch.setattr(pipeline, "extract_into_build", provider_must_not_run)
    monkeypatch.setattr(coreference, "resolve_document_mentions", provider_must_not_run)
    monkeypatch.setattr(
        persistence, "persist_document_resolution", provider_must_not_run
    )

    activated = builds.build_document_graph(
        document.id,
        document.full_text_hash,
        builds.derive_document_build_key(context.identity),
    )

    artifact.refresh_from_db()
    run.refresh_from_db()
    assert activated.pk == artifact.pk
    assert artifact.status == GraphArtifact.Status.ACTIVE
    assert run.stage == GraphBuildRun.Stage.ACTIVE
    assert run.attempt == 2


def test_post_assembly_resume_skips_providers_and_activates_same_collection(
    monkeypatch,
):
    from apps.collections.models import Collection
    from apps.knowledge_graph.graph import assembly, filtering
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.resolution import collection as resolution_collection
    from apps.knowledge_graph.services import builds

    collection = Collection.objects.create(name=f"resume {uuid.uuid4().hex}")
    context = _collection_context(collection)
    artifact, run, owner, lease_generation = _collection_occurrence(
        context,
        artifact_status=GraphArtifact.Status.BUILDING,
        run_stage=GraphBuildRun.Stage.QUEUED,
        run_status=GraphBuildRun.Status.PENDING,
        attempt=2,
        claim=True,
    )
    run.stats = {"fixture_commit_state": "validated_by_injected_inspectors"}
    run.save(update_fields=["stats"])
    monkeypatch.setattr(builds, "_collection_context", lambda *_args: context)
    monkeypatch.setattr(
        builds,
        "_bootstrap_collection_build",
        lambda *_args: (artifact, run, owner, lease_generation, False),
    )
    monkeypatch.setattr(
        builds,
        "_collection_resolution_commit_state",
        lambda *_args, **_kwargs: builds.CommitMarkerState.VALID,
    )
    monkeypatch.setattr(
        builds,
        "_collection_assembly_commit_state",
        lambda *_args, **_kwargs: builds.CommitMarkerState.VALID,
    )
    monkeypatch.setattr(
        assembly,
        "_validate_locked_complete_artifact",
        lambda **_kwargs: SimpleNamespace(),
    )

    def provider_must_not_run(*_args, **_kwargs):
        pytest.fail("a committed pre-activation resume called a provider")

    for module, name in (
        (resolution_collection, "load_collection_resolution_inputs"),
        (resolution_collection, "resolve_collection_entities"),
        (resolution_collection, "load_collection_filter_inputs"),
        (resolution_collection, "persist_collection_resolution"),
        (filtering, "filter_collection_resolution"),
        (assembly, "assemble_collection_graph"),
    ):
        monkeypatch.setattr(module, name, provider_must_not_run)

    activated = builds.refresh_collection_graph(
        collection.pk,
        context.identity.aggregate_source_signature,
        builds.derive_collection_build_key(context.identity),
    )

    artifact.refresh_from_db()
    run.refresh_from_db()
    assert activated.pk == artifact.pk
    assert artifact.status == GraphArtifact.Status.ACTIVE
    assert run.stage == GraphBuildRun.Stage.ACTIVE
    assert run.attempt == 2


def test_document_refresh_on_commit_is_discarded_by_outer_rollback(monkeypatch):
    from apps.knowledge_graph.services import builds

    collection, document, chunk = _persist_document(label="on-commit-rollback")
    context = _document_context(document, chunk)
    callbacks = []
    monkeypatch.setattr(builds, "_enqueue_current_collection_refresh", callbacks.append)

    with pytest.raises(RuntimeError, match="rollback injection"):
        with transaction.atomic():
            builds._register_document_refresh_callbacks(
                context,
                SimpleNamespace(metadata={"initial_collection_id": collection.pk}),
            )
            raise RuntimeError("rollback injection")

    assert callbacks == []


def test_collection_lock_paths_complete_without_a_deadlock():
    from apps.collections.models import Collection
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services import builds

    collection = Collection.objects.create(name=f"lock order {uuid.uuid4().hex}")
    context = _collection_context(collection)
    artifact, run, owner, lease_generation = _collection_occurrence(
        context,
        artifact_status=GraphArtifact.Status.BUILDING,
        run_stage=GraphBuildRun.Stage.QUEUED,
        run_status=GraphBuildRun.Status.PENDING,
        claim=True,
    )
    build_key = builds.derive_collection_build_key(context.identity)
    barrier = Barrier(2)

    def coordinator_lock():
        close_old_connections()
        try:
            with transaction.atomic():
                barrier.wait(timeout=10)
                _collection, artifacts, runs = builds._lock_collection_build_rows(
                    collection.pk,
                    build_key=build_key,
                    candidate_artifact_id=artifact.pk,
                )
                return tuple(row.pk for row in artifacts), tuple(row.pk for row in runs)
        finally:
            close_old_connections()

    def assembly_lock():
        close_old_connections()
        try:
            with transaction.atomic():
                barrier.wait(timeout=10)
                _collection, candidate, candidate_run, _scope = (
                    assembly._locked_candidate(
                        collection.pk,
                        run.pk,
                        lock_competing_runs=True,
                        lease_owner=owner,
                        lease_generation=lease_generation,
                    )
                )
                return candidate.pk, candidate_run.pk
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        coordinator = executor.submit(coordinator_lock)
        assembler = executor.submit(assembly_lock)
        coordinator_ids = coordinator.result(timeout=30)
        assembly_ids = assembler.result(timeout=30)

    assert artifact.pk in coordinator_ids[0]
    assert run.pk in coordinator_ids[1]
    assert assembly_ids == (artifact.pk, run.pk)


def test_database_clock_takeover_heartbeat_and_lost_token_ignore_app_clock(
    monkeypatch,
):
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services import builds

    collection = Collection.objects.create(name=f"db clock {uuid.uuid4().hex}")
    context = _collection_context(collection)
    _artifact, run, old_owner, old_generation = _collection_occurrence(
        context,
        artifact_status=GraphArtifact.Status.BUILDING,
        run_stage=GraphBuildRun.Stage.QUEUED,
        run_status=GraphBuildRun.Status.PENDING,
        claim=True,
    )
    monkeypatch.setattr(
        builds.timezone,
        "now",
        lambda: datetime(2099, 1, 1, tzinfo=UTC),
    )
    assert builds._run_has_live_lease(run)

    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE apps_knowledge_graph_graphbuildrun "
            "SET lease_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 second' "
            "WHERE id = %s",
            [run.pk],
        )
    run.refresh_from_db()
    with transaction.atomic():
        locked = GraphBuildRun.objects.select_for_update().get(pk=run.pk)
        new_owner, new_generation = builds._claim_locked_run(locked, "worker-new")
    assert new_generation == old_generation + 1

    with pytest.raises(builds.BuildLeaseLostError):
        builds.renew_build_lease(run.pk, old_owner, old_generation)
    run.refresh_from_db()
    before_heartbeat = run.lease_expires_at
    with builds.LeaseHeartbeat(
        run.pk,
        new_owner,
        new_generation,
        interval_seconds=60,
    ):
        pass
    run.refresh_from_db()
    assert run.lease_expires_at >= before_heartbeat
    with pytest.raises(builds.BuildLeaseLostError):
        builds.validate_build_lease(run, old_owner, old_generation)


def test_terminalization_survives_document_and_collection_deletion():
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services import builds

    collection, document, chunk = _persist_document(label="deleted-document")
    document_context = _document_context(document, chunk)
    doc_artifact, doc_run, doc_owner, doc_generation = _document_occurrence(
        document_context,
        generation=1,
        artifact_status=GraphArtifact.Status.BUILDING,
        run_stage=GraphBuildRun.Stage.EXTRACTING,
        run_status=GraphBuildRun.Status.RUNNING,
        claim=True,
    )
    document.delete()
    builds._terminal_document_build(
        document_context,
        doc_artifact.pk,
        doc_run.pk,
        lease_owner=doc_owner,
        lease_generation=doc_generation,
        stale=False,
        error_code="source_deleted",
    )
    doc_artifact.refresh_from_db()
    doc_run.refresh_from_db()
    assert doc_artifact.status == GraphArtifact.Status.FAILED
    assert doc_run.stage == GraphBuildRun.Stage.FAILED

    deleted_collection = Collection.objects.create(
        name=f"deleted collection {uuid.uuid4().hex}"
    )
    collection_context = _collection_context(deleted_collection, source_digit="d")
    col_artifact, col_run, col_owner, col_generation = _collection_occurrence(
        collection_context,
        artifact_status=GraphArtifact.Status.BUILDING,
        run_stage=GraphBuildRun.Stage.RESOLVING,
        run_status=GraphBuildRun.Status.RUNNING,
        claim=True,
    )
    deleted_collection.delete()
    builds._terminal_collection_build(
        collection_context,
        col_artifact.pk,
        col_run.pk,
        lease_owner=col_owner,
        lease_generation=col_generation,
        stale=False,
        error_code="source_deleted",
    )
    col_artifact.refresh_from_db()
    col_run.refresh_from_db()
    assert col_artifact.status == GraphArtifact.Status.FAILED
    assert col_run.stage == GraphBuildRun.Stage.FAILED
