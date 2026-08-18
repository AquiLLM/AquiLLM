from __future__ import annotations

import inspect
import os
import socket
import uuid
from dataclasses import fields, replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.utils import timezone

DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")


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


def _document_identity():
    from apps.knowledge_graph.services.builds import DocumentBuildIdentity

    return DocumentBuildIdentity(
        document_id=DOCUMENT_ID,
        source_hash="a" * 64,
        ordered_chunk_signature="b" * 64,
        extractor_package="gliner2==1.3.2",
        extractor_checkpoint="fastino/gliner2-base-v1",
        extractor_model_revision="c" * 40,
        extractor_config_checksum="9" * 64,
        ontology_version="1.0.0",
        ontology_checksum="d" * 64,
        resolver_version="document-coreference-v1",
        resolver_checksum="e" * 64,
        filter_version="unfiltered-v1",
        filter_checksum="f" * 64,
        assembly_version="not-applicable",
        assembly_checksum="1" * 64,
    )


def _collection_identity():
    from apps.knowledge_graph.services.builds import CollectionBuildIdentity

    return CollectionBuildIdentity(
        collection_id=17,
        aggregate_source_signature="a" * 64,
        extractor_version="gliner2:checkpoint@revision",
        ontology_version="1.0.0",
        ontology_checksum="b" * 64,
        resolver_version="collection-resolution-v1",
        resolver_checksum="c" * 64,
        filter_version="collection-filter-v1",
        filter_checksum="d" * 64,
        assembly_version="collection-assembly-v1",
        assembly_checksum="e" * 64,
        embedding_model_signature=(
            f"local:model@revision:endpoint={'f' * 64}:dims=1024:"
            "prep=kg-entity-v1:max_chars=8192:batch=64"
        ),
    )


def test_public_build_entrypoints_keep_the_task_contract_narrow():
    from apps.knowledge_graph.services.builds import (
        build_document_graph,
        refresh_collection_graph,
    )

    assert tuple(inspect.signature(build_document_graph).parameters) == (
        "document_id",
        "expected_source_hash",
        "document_build_key",
    )
    assert tuple(inspect.signature(refresh_collection_graph).parameters) == (
        "collection_id",
        "aggregate_source_signature",
        "collection_build_key",
    )


def test_document_build_key_binds_every_immutable_input():
    from apps.knowledge_graph.services.builds import derive_document_build_key

    identity = _document_identity()
    original = derive_document_build_key(identity)
    assert len(original) == 64
    assert original == derive_document_build_key(identity)

    replacements = {
        "document_id": uuid.UUID("22222222-2222-4222-8222-222222222222"),
        "source_hash": "2" * 64,
        "ordered_chunk_signature": "3" * 64,
        "extractor_package": "gliner2==9.9.9",
        "extractor_checkpoint": "other/checkpoint",
        "extractor_model_revision": "4" * 40,
        "extractor_config_checksum": "0" * 64,
        "ontology_version": "2.0.0",
        "ontology_checksum": "5" * 64,
        "resolver_version": "resolver-v2",
        "resolver_checksum": "6" * 64,
        "filter_version": "filter-v2",
        "filter_checksum": "7" * 64,
        "assembly_version": "assembly-v2",
        "assembly_checksum": "8" * 64,
        "ontology_activation_signature": "a" * 64,
    }
    for field, changed in replacements.items():
        assert (
            derive_document_build_key(replace(identity, **{field: changed})) != original
        )


def test_collection_build_key_binds_manifest_and_all_policy_identities():
    from apps.knowledge_graph.services.builds import derive_collection_build_key

    identity = _collection_identity()
    original = derive_collection_build_key(identity)
    replacements = {
        "collection_id": 18,
        "aggregate_source_signature": "1" * 64,
        "extractor_version": "extractor-v2",
        "ontology_version": "2.0.0",
        "ontology_checksum": "2" * 64,
        "resolver_version": "resolver-v2",
        "resolver_checksum": "3" * 64,
        "filter_version": "filter-v2",
        "filter_checksum": "4" * 64,
        "assembly_version": "assembly-v2",
        "assembly_checksum": "5" * 64,
        "embedding_model_signature": (
            f"local:model@revision:endpoint={'6' * 64}:dims=1024:"
            "prep=kg-entity-v1:max_chars=8192:batch=64"
        ),
        "ontology_activation_signature": "7" * 64,
    }
    assert len(original) == 64
    for field, changed in replacements.items():
        assert (
            derive_collection_build_key(replace(identity, **{field: changed}))
            != original
        )


def test_document_and_collection_transitions_are_separate_and_cannot_skip():
    from apps.knowledge_graph.models import GraphBuildRun
    from apps.knowledge_graph.services.builds import validate_stage_transition

    document_path = (
        GraphBuildRun.Stage.QUEUED,
        GraphBuildRun.Stage.EXTRACTING,
        GraphBuildRun.Stage.RESOLVING,
        GraphBuildRun.Stage.VALIDATING,
        GraphBuildRun.Stage.ACTIVE,
        GraphBuildRun.Stage.SUPERSEDED,
    )
    collection_path = (
        GraphBuildRun.Stage.QUEUED,
        GraphBuildRun.Stage.SNAPSHOTTING,
        GraphBuildRun.Stage.RESOLVING,
        GraphBuildRun.Stage.ASSEMBLING,
        GraphBuildRun.Stage.VALIDATING,
        GraphBuildRun.Stage.ACTIVE,
        GraphBuildRun.Stage.STALE,
    )
    for left, right in zip(document_path, document_path[1:], strict=False):
        validate_stage_transition(GraphBuildRun.BuildKind.DOCUMENT, left, right)
    for left, right in zip(collection_path, collection_path[1:], strict=False):
        validate_stage_transition(GraphBuildRun.BuildKind.COLLECTION, left, right)

    with pytest.raises(ValidationError, match="transition"):
        validate_stage_transition(
            GraphBuildRun.BuildKind.DOCUMENT,
            GraphBuildRun.Stage.QUEUED,
            GraphBuildRun.Stage.ACTIVE,
        )
    with pytest.raises(ValidationError, match="transition"):
        validate_stage_transition(
            GraphBuildRun.BuildKind.DOCUMENT,
            GraphBuildRun.Stage.QUEUED,
            GraphBuildRun.Stage.SNAPSHOTTING,
        )
    with pytest.raises(ValidationError, match="transition"):
        validate_stage_transition(
            GraphBuildRun.BuildKind.COLLECTION,
            GraphBuildRun.Stage.QUEUED,
            GraphBuildRun.Stage.EXTRACTING,
        )


def test_graph_build_run_has_typed_orchestration_and_durable_lease_fields():
    from apps.knowledge_graph.models import GraphBuildRun

    stages = {value for value, _label in GraphBuildRun.Stage.choices}
    assert {
        "queued",
        "extracting",
        "snapshotting",
        "resolving",
        "assembling",
        "validating",
        "active",
        "failed",
        "superseded",
        "stale",
    } <= stages
    assert GraphBuildRun._meta.get_field("build_key").max_length == 64
    assert GraphBuildRun._meta.get_field("lease_generation").default == 0
    assert GraphBuildRun._meta.get_field("lease_owner").max_length == 128
    assert GraphBuildRun._meta.get_field("lease_expires_at").null is True
    assert (
        GraphBuildRun._meta.get_field("stage_marker").get_internal_type() == "JSONField"
    )
    constraint_names = {item.name for item in GraphBuildRun._meta.constraints}
    assert {
        "kg_build_identity_unique",
        "kg_build_kind_matches_scope",
        "kg_build_stage_matches_kind",
        "kg_build_stage_status_valid",
        "kg_build_lease_complete",
        "kg_build_terminal_lease_clear",
    } <= constraint_names
    index_names = {item.name for item in GraphBuildRun._meta.indexes}
    assert "kg_run_status_lease_idx" in index_names


def test_every_orchestrated_mutation_requires_the_exact_live_lease_generation():
    from apps.knowledge_graph.services.builds import (
        BuildLeaseLostError,
        validate_build_lease,
    )

    now = timezone.now()
    run = SimpleNamespace(
        metadata={"orchestration_version": 1},
        lease_owner="worker-a",
        lease_generation=4,
        lease_expires_at=now + timedelta(minutes=2),
    )
    validate_build_lease(run, "worker-a", 4, now=now)
    with pytest.raises(BuildLeaseLostError, match="owner"):
        validate_build_lease(run, "worker-b", 4, now=now)
    with pytest.raises(BuildLeaseLostError, match="generation"):
        validate_build_lease(run, "worker-a", 3, now=now)
    with pytest.raises(BuildLeaseLostError, match="expired"):
        validate_build_lease(run, "worker-a", 4, now=now + timedelta(minutes=3))

    legacy = SimpleNamespace(metadata={}, lease_owner="", lease_generation=0)
    validate_build_lease(legacy, None, None, now=now)


def test_mutating_task_seams_accept_owner_and_generation_fences():
    from apps.knowledge_graph.extraction.pipeline import extract_into_build
    from apps.knowledge_graph.graph.assembly import (
        activate_collection_graph,
        assemble_collection_graph,
        validate_collection_graph_artifact,
    )
    from apps.knowledge_graph.resolution.collection import (
        load_collection_filter_inputs,
        load_collection_resolution_inputs,
        persist_collection_resolution,
    )
    from apps.knowledge_graph.resolution.persistence import (
        persist_document_resolution,
    )

    for function in (
        extract_into_build,
        persist_document_resolution,
        load_collection_resolution_inputs,
        load_collection_filter_inputs,
        persist_collection_resolution,
        assemble_collection_graph,
        validate_collection_graph_artifact,
        activate_collection_graph,
    ):
        parameters = inspect.signature(function).parameters
        assert "lease_owner" in parameters, function.__name__
        assert "lease_generation" in parameters, function.__name__


def test_build_identity_rejects_noncanonical_or_unbounded_values():
    from apps.knowledge_graph.services.builds import DocumentBuildIdentity

    with pytest.raises(ValueError, match="source hash"):
        replace(_document_identity(), source_hash="not-a-hash")
    with pytest.raises(ValueError, match="extractor package"):
        replace(_document_identity(), extractor_package=" x ")
    with pytest.raises(ValueError, match="document UUID"):
        identity = _document_identity()
        DocumentBuildIdentity(
            **{
                **{
                    field.name: getattr(identity, field.name)
                    for field in fields(identity)
                },
                "document_id": "not-a-uuid",
            }
        )


def test_same_text_rechunk_and_concrete_model_change_document_build_key():
    from apps.knowledge_graph.services.builds import (
        derive_document_build_key,
        ordered_chunk_signature,
    )

    def chunk(pk: int):
        return SimpleNamespace(
            pk=pk,
            doc_id=DOCUMENT_ID,
            chunk_number=0,
            start_position=0,
            end_position=4,
            modality="text",
            content="same",
        )

    first_signature = ordered_chunk_signature(
        (chunk(10),), concrete_model_label="apps_documents.rawtextdocument"
    )
    rechunked_signature = ordered_chunk_signature(
        (chunk(11),), concrete_model_label="apps_documents.rawtextdocument"
    )
    other_model_signature = ordered_chunk_signature(
        (chunk(10),), concrete_model_label="apps_documents.pdfdocument"
    )
    assert len({first_signature, rechunked_signature, other_model_signature}) == 3
    identity = _document_identity()
    assert derive_document_build_key(
        replace(identity, ordered_chunk_signature=first_signature)
    ) != derive_document_build_key(
        replace(identity, ordered_chunk_signature=rechunked_signature)
    )


def test_document_refresh_callback_resolves_an_exact_post_commit_key(monkeypatch):
    from apps.knowledge_graph.services import builds

    context = SimpleNamespace(identity=_collection_identity())
    scheduled = []
    monkeypatch.setattr(builds, "_collection_context", lambda collection_id: context)
    monkeypatch.setattr(
        builds,
        "enqueue_collection_refresh",
        lambda collection_id, aggregate, build_key: scheduled.append(
            (collection_id, aggregate, build_key)
        ),
    )

    builds._enqueue_current_collection_refresh(context.identity.collection_id)

    assert scheduled == [
        (
            context.identity.collection_id,
            context.identity.aggregate_source_signature,
            builds.derive_collection_build_key(context.identity),
        )
    ]


def test_document_move_registers_both_collection_refreshes_on_commit(monkeypatch):
    from apps.knowledge_graph.services import builds

    callbacks = []
    refreshed = []
    current = builds._DocumentContext(
        identity=_document_identity(),
        collection_id=18,
        ontology=SimpleNamespace(),
        settings=SimpleNamespace(),
    )
    run = SimpleNamespace(metadata={"initial_collection_id": 17})
    monkeypatch.setattr(builds.transaction, "on_commit", callbacks.append)
    monkeypatch.setattr(
        builds,
        "_enqueue_current_collection_refresh",
        refreshed.append,
    )

    builds._register_document_refresh_callbacks(current, run)

    assert refreshed == []
    assert len(callbacks) == 2
    for callback in callbacks:
        callback()
    assert refreshed == [17, 18]


def _orchestration_artifact(*, build_key: str = "7" * 64):
    from apps.knowledge_graph.models import GraphArtifact

    return GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        build_key=build_key,
        status=GraphArtifact.Status.BUILDING,
        source_hash="a" * 64,
        ontology_version="research-v1",
        extractor_version="gliner2:model@revision",
        resolver_version="document-coreference-v1",
        filter_policy_version="pending-v1",
        metadata={"orchestration_version": 1},
    )


def _orchestration_run(artifact):
    from apps.knowledge_graph.models import GraphBuildRun

    return GraphBuildRun.objects.create(
        artifact=artifact,
        stage=GraphBuildRun.Stage.QUEUED,
        status=GraphBuildRun.Status.PENDING,
        attempt=1,
        metadata={"orchestration_version": 1, "attempt_history": []},
    )


@pytest.mark.django_db(transaction=True)
@database_required
def test_expired_takeover_reuses_one_logical_run_and_fences_old_worker():
    from apps.knowledge_graph.models import GraphBuildRun
    from apps.knowledge_graph.services.builds import (
        BuildLeaseLostError,
        _claim_locked_run,
        _restart_locked_run,
        _transition_run,
    )

    artifact = _orchestration_artifact()
    run = _orchestration_run(artifact)
    with transaction.atomic():
        locked = GraphBuildRun.objects.select_for_update().get(pk=run.pk)
        old_owner, old_generation = _claim_locked_run(locked, "worker-old")
    GraphBuildRun.objects.filter(pk=run.pk).update(
        lease_expires_at=timezone.now() - timedelta(seconds=1)
    )
    with transaction.atomic():
        locked = GraphBuildRun.objects.select_for_update().get(pk=run.pk)
        _restart_locked_run(locked)
        new_owner, new_generation = _claim_locked_run(locked, "worker-new")

    assert old_generation == 1
    assert new_generation == 2
    assert (
        GraphBuildRun.objects.filter(
            build_kind=GraphBuildRun.BuildKind.DOCUMENT,
            scope_id=str(DOCUMENT_ID),
            build_key=artifact.build_key,
        ).count()
        == 1
    )
    run.refresh_from_db()
    assert run.attempt == 2
    with pytest.raises(BuildLeaseLostError):
        _transition_run(
            run.pk,
            GraphBuildRun.Stage.FAILED,
            lease_owner=old_owner,
            lease_generation=old_generation,
        )
    run.refresh_from_db()
    assert (run.stage, run.status) == (
        GraphBuildRun.Stage.QUEUED,
        GraphBuildRun.Status.PENDING,
    )
    _transition_run(
        run.pk,
        GraphBuildRun.Stage.FAILED,
        lease_owner=new_owner,
        lease_generation=new_generation,
    )
    run.refresh_from_db()
    assert (run.stage, run.status) == (
        GraphBuildRun.Stage.FAILED,
        GraphBuildRun.Status.FAILED,
    )


@pytest.mark.django_db(transaction=True)
@database_required
def test_persistence_rejects_a_second_logical_run_for_the_same_build_key():
    artifact = _orchestration_artifact()
    _orchestration_run(artifact)

    with pytest.raises((ValidationError, IntegrityError)), transaction.atomic():
        _orchestration_run(artifact)


def _stage_transition_stub(run):
    def transition(
        run_id,
        target,
        *,
        lease_owner,
        lease_generation,
        marker=None,
    ):
        assert run_id == run.pk
        assert (lease_owner, lease_generation) == ("owner", 3)
        run.stage = target
        return run

    return transition


def test_duplicate_document_delivery_returns_the_exact_committed_artifact(
    monkeypatch,
):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.services import builds

    identity = _document_identity()
    context = builds._DocumentContext(
        identity=identity,
        collection_id=17,
        ontology=SimpleNamespace(),
        settings=SimpleNamespace(),
    )
    artifact = SimpleNamespace(
        pk=40, build_key=builds.derive_document_build_key(identity)
    )
    run = SimpleNamespace(pk=50)
    monkeypatch.setattr(builds, "_document_context", lambda *_args: context)
    monkeypatch.setattr(
        builds,
        "_bootstrap_document_build",
        lambda *_args: (artifact, run, None, None, True),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_into_build",
        lambda *_args, **_kwargs: pytest.fail("duplicate repeated extraction"),
    )

    result = builds.build_document_graph(
        identity.document_id,
        identity.source_hash,
        artifact.build_key,
    )

    assert result is artifact


def test_provider_failure_terminals_only_the_candidate_document_build(monkeypatch):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import GraphBuildRun
    from apps.knowledge_graph.services import builds

    identity = _document_identity()
    context = builds._DocumentContext(
        identity=identity,
        collection_id=17,
        ontology=SimpleNamespace(),
        settings=SimpleNamespace(),
    )
    artifact = SimpleNamespace(pk=41)
    run = SimpleNamespace(
        pk=51,
        stage=GraphBuildRun.Stage.QUEUED,
        attempt=1,
        stats={},
    )
    terminal_calls = []
    monkeypatch.setattr(builds, "_document_context", lambda *_args: context)
    monkeypatch.setattr(
        builds,
        "_bootstrap_document_build",
        lambda *_args: (artifact, run, "owner", 3, False),
    )
    monkeypatch.setattr(builds, "_transition_run", _stage_transition_stub(run))
    monkeypatch.setattr(
        pipeline,
        "extract_into_build",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("provider unavailable")
        ),
    )
    monkeypatch.setattr(
        builds,
        "_terminal_document_build",
        lambda context, artifact_id, run_id, **kwargs: terminal_calls.append(
            (artifact_id, run_id, kwargs["stale"], kwargs["error_code"])
        ),
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        builds.build_document_graph(
            identity.document_id,
            identity.source_hash,
            builds.derive_document_build_key(identity),
        )

    assert terminal_calls == [(41, 51, False, "document_build_failed")]


def test_midflight_hash_change_is_a_stale_document_terminal(monkeypatch):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import GraphBuildRun
    from apps.knowledge_graph.services import builds

    identity = _document_identity()
    context = builds._DocumentContext(
        identity=identity,
        collection_id=17,
        ontology=SimpleNamespace(),
        settings=SimpleNamespace(),
    )
    artifact = SimpleNamespace(pk=42)
    run = SimpleNamespace(
        pk=52,
        stage=GraphBuildRun.Stage.QUEUED,
        attempt=1,
        stats={},
    )
    terminal_calls = []
    monkeypatch.setattr(builds, "_document_context", lambda *_args: context)
    monkeypatch.setattr(
        builds,
        "_bootstrap_document_build",
        lambda *_args: (artifact, run, "owner", 3, False),
    )
    monkeypatch.setattr(builds, "_transition_run", _stage_transition_stub(run))
    monkeypatch.setattr(
        pipeline,
        "extract_into_build",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            pipeline.MidflightSourceChangedError("changed")
        ),
    )
    monkeypatch.setattr(
        builds,
        "_terminal_document_build",
        lambda context, artifact_id, run_id, **kwargs: terminal_calls.append(
            (kwargs["stale"], kwargs["error_code"])
        ),
    )

    with pytest.raises(pipeline.MidflightSourceChangedError):
        builds.build_document_graph(
            identity.document_id,
            identity.source_hash,
            builds.derive_document_build_key(identity),
        )

    assert terminal_calls == [(True, "source_or_config_stale")]


def test_document_retry_uses_commit_markers_without_repeating_provider_work(
    monkeypatch,
):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import GraphBuildRun
    from apps.knowledge_graph.resolution import coreference
    from apps.knowledge_graph.services import builds

    identity = _document_identity()
    context = builds._DocumentContext(
        identity=identity,
        collection_id=17,
        ontology=SimpleNamespace(),
        settings=SimpleNamespace(),
    )
    artifact = SimpleNamespace(pk=43)
    run = SimpleNamespace(
        pk=53,
        stage=GraphBuildRun.Stage.QUEUED,
        attempt=2,
        stats={"extraction_commit": {}, "resolution_commit": {}},
    )
    activated = SimpleNamespace(pk=43)
    monkeypatch.setattr(builds, "_document_context", lambda *_args: context)
    monkeypatch.setattr(
        builds,
        "_bootstrap_document_build",
        lambda *_args: (artifact, run, "owner", 3, False),
    )
    monkeypatch.setattr(builds, "_transition_run", _stage_transition_stub(run))
    monkeypatch.setattr(
        pipeline,
        "extract_into_build",
        lambda *_args, **_kwargs: pytest.fail("extraction provider repeated"),
    )
    monkeypatch.setattr(
        coreference,
        "resolve_document_mentions",
        lambda *_args, **_kwargs: pytest.fail("document resolution repeated"),
    )
    monkeypatch.setattr(
        builds,
        "_activate_document_build",
        lambda *_args, **_kwargs: (activated, {"entity_mention_count": 2}),
    )

    result = builds.build_document_graph(
        identity.document_id,
        identity.source_hash,
        builds.derive_document_build_key(identity),
    )

    assert result is activated
    assert run.stage == GraphBuildRun.Stage.VALIDATING


def _collection_context_for(identity):
    from apps.knowledge_graph.services import builds

    return builds._CollectionContext(
        identity=identity,
        collection=SimpleNamespace(pk=identity.collection_id),
        document_artifacts=(),
        ontology=SimpleNamespace(),
        filter_policy=SimpleNamespace(),
        resolution_config=SimpleNamespace(),
        assembly_config=SimpleNamespace(),
    )


def test_collection_retry_after_assembly_skips_embedding_and_validates(monkeypatch):
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.resolution import collection as resolution
    from apps.knowledge_graph.services import builds

    identity = _collection_identity()
    context = _collection_context_for(identity)
    artifact = SimpleNamespace(pk=61)
    run = SimpleNamespace(
        pk=71,
        stage=GraphBuildRun.Stage.QUEUED,
        attempt=2,
        stats={
            "collection_resolution_commit": {},
            "collection_assembly_commit": {},
        },
    )
    calls = []
    monkeypatch.setattr(builds, "_collection_context", lambda *_args: context)
    monkeypatch.setattr(
        builds,
        "_bootstrap_collection_build",
        lambda *_args: (artifact, run, "owner", 3, False),
    )
    monkeypatch.setattr(builds, "_transition_run", _stage_transition_stub(run))
    monkeypatch.setattr(
        resolution,
        "resolve_collection_entities",
        lambda *_args, **_kwargs: pytest.fail("embedding resolution repeated"),
    )
    monkeypatch.setattr(
        assembly,
        "assemble_collection_graph",
        lambda *_args, **_kwargs: calls.append("assemble"),
    )
    monkeypatch.setattr(
        assembly,
        "validate_collection_graph_artifact",
        lambda *_args, **_kwargs: calls.append("validate"),
    )
    monkeypatch.setattr(
        assembly,
        "activate_collection_graph",
        lambda *_args, **_kwargs: calls.append("activate"),
    )
    monkeypatch.setattr(GraphArtifact.objects, "get", lambda **_kwargs: artifact)
    monkeypatch.setattr(GraphBuildRun.objects, "get", lambda **_kwargs: run)

    result = builds.refresh_collection_graph(
        identity.collection_id,
        identity.aggregate_source_signature,
        builds.derive_collection_build_key(identity),
    )

    assert result is artifact
    assert calls == ["assemble", "validate", "activate"]


def test_collection_policy_drift_with_same_manifest_is_stale_and_rescheduled(
    monkeypatch,
):
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.models import GraphBuildRun
    from apps.knowledge_graph.services import builds

    initial_identity = _collection_identity()
    current_identity = replace(
        initial_identity,
        ontology_activation_signature="8" * 64,
    )
    contexts = iter(
        (
            _collection_context_for(initial_identity),
            _collection_context_for(current_identity),
        )
    )
    artifact = SimpleNamespace(pk=62)
    run = SimpleNamespace(
        pk=72,
        stage=GraphBuildRun.Stage.VALIDATING,
        attempt=1,
        stats={"collection_resolution_commit": {}},
        refresh_from_db=lambda: None,
    )
    terminal_calls = []
    monkeypatch.setattr(builds, "_collection_context", lambda *_args: next(contexts))
    monkeypatch.setattr(
        builds,
        "_bootstrap_collection_build",
        lambda *_args: (artifact, run, "owner", 3, False),
    )
    monkeypatch.setattr(
        assembly,
        "validate_collection_graph_artifact",
        lambda *_args, **_kwargs: None,
    )

    def terminal(*_args, **kwargs):
        terminal_calls.append((kwargs["stale"], kwargs["reschedule"]))
        run.stage = GraphBuildRun.Stage.STALE

    monkeypatch.setattr(builds, "_terminal_collection_build", terminal)

    with pytest.raises(builds.StaleBuildError):
        builds.refresh_collection_graph(
            initial_identity.collection_id,
            initial_identity.aggregate_source_signature,
            builds.derive_collection_build_key(initial_identity),
        )

    assert terminal_calls == [(True, True)]


def test_stale_same_key_collection_prerequisite_does_not_reschedule_itself(
    monkeypatch,
):
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.models import GraphBuildRun
    from apps.knowledge_graph.services import builds

    identity = _collection_identity()
    context = _collection_context_for(identity)
    artifact = SimpleNamespace(pk=63)
    run = SimpleNamespace(
        pk=73,
        stage=GraphBuildRun.Stage.ASSEMBLING,
        attempt=1,
        stats={"collection_resolution_commit": {}},
    )
    terminal_calls = []
    monkeypatch.setattr(builds, "_collection_context", lambda *_args: context)
    monkeypatch.setattr(
        builds,
        "_bootstrap_collection_build",
        lambda *_args: (artifact, run, "owner", 3, False),
    )
    monkeypatch.setattr(
        assembly,
        "assemble_collection_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            assembly.CollectionGraphSourceStaleError("source stale")
        ),
    )
    monkeypatch.setattr(
        builds,
        "_terminal_collection_build",
        lambda *_args, **kwargs: terminal_calls.append(
            (kwargs["stale"], kwargs["reschedule"])
        ),
    )

    with pytest.raises(assembly.CollectionGraphSourceStaleError):
        builds.refresh_collection_graph(
            identity.collection_id,
            identity.aggregate_source_signature,
            builds.derive_collection_build_key(identity),
        )

    assert terminal_calls == [(True, False)]
