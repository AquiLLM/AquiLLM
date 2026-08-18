from __future__ import annotations

import importlib
import inspect
import os
import socket
import uuid
from contextlib import nullcontext
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
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

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
    assert GraphArtifact._meta.get_field("build_generation").default == 1
    assert GraphBuildRun._meta.get_field("build_generation").default == 1
    assert GraphArtifact._meta.get_field("orchestration_version").default == 0
    assert GraphBuildRun._meta.get_field("orchestration_version").default == 0
    assert "build_generation" in GraphArtifact._IMMUTABLE_FIELDS
    assert "orchestration_version" in GraphArtifact._IMMUTABLE_FIELDS
    assert "build_generation" in GraphBuildRun._IMMUTABLE_FIELDS
    assert "orchestration_version" in GraphBuildRun._IMMUTABLE_FIELDS
    assert GraphBuildRun._meta.get_field("lease_generation").default == 0
    assert GraphBuildRun._meta.get_field("lease_owner").max_length == 128
    assert GraphBuildRun._meta.get_field("lease_expires_at").null is True
    assert (
        GraphBuildRun._meta.get_field("stage_marker").get_internal_type() == "JSONField"
    )
    constraint_names = {item.name for item in GraphBuildRun._meta.constraints}
    assert {
        "kg_run_artifact_occurrence_unique",
        "kg_build_occurrence_unique",
        "kg_build_kind_matches_scope",
        "kg_build_generation_positive",
        "kg_build_orchestration_version_valid",
        "kg_build_stage_matches_kind",
        "kg_build_stage_status_valid",
        "kg_build_lease_complete",
        "kg_build_terminal_lease_clear",
    } <= constraint_names
    artifact_constraint_names = {item.name for item in GraphArtifact._meta.constraints}
    assert {
        "kg_artifact_build_occurrence",
        "kg_artifact_generation_positive",
        "kg_artifact_orchestration_version_valid",
    } <= artifact_constraint_names
    index_names = {item.name for item in GraphBuildRun._meta.indexes}
    assert "kg_run_status_lease_idx" in index_names


def test_task7_and_task8_use_the_shared_artifact_orchestration_discriminator():
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.resolution import persistence
    from apps.knowledge_graph.services import builds

    assert hasattr(GraphArtifact, "OrchestrationVersion")
    assert not hasattr(GraphBuildRun, "OrchestrationVersion")
    for function in (
        pipeline._find_committed_extraction_run,
        persistence.persist_document_resolution,
        builds._document_extraction_commit_state,
    ):
        assert "GraphBuildRun.OrchestrationVersion" not in inspect.getsource(function)


def test_task11_migration_streams_occurrence_backfill_without_an_all_row_map():
    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0002_graph_build_run_stages"
    )
    source = inspect.getsource(migration.populate_build_keys)

    assert ".iterator(" in source
    assert "bulk_update" in source
    assert "artifact_keys" not in source
    operation_text = "\n".join(map(repr, migration.Migration.operations))
    assert "build_generation" in operation_text
    assert "orchestration_version" in operation_text


def test_every_orchestrated_mutation_requires_the_exact_live_lease_generation():
    from apps.knowledge_graph.services.builds import (
        BuildLeaseLostError,
        validate_build_lease,
    )

    run = SimpleNamespace(
        orchestration_version=1,
        lease_owner="worker-a",
        lease_generation=4,
    )
    with pytest.raises(BuildLeaseLostError, match="owner"):
        validate_build_lease(run, "worker-b", 4)
    with pytest.raises(BuildLeaseLostError, match="generation"):
        validate_build_lease(run, "worker-a", 3)
    with pytest.raises(BuildLeaseLostError, match="persisted"):
        validate_build_lease(run, "worker-a", 4)

    legacy = SimpleNamespace(
        orchestration_version=0,
        lease_owner="",
        lease_generation=0,
    )
    validate_build_lease(legacy, None, None)


def test_lease_claim_validation_and_renewal_use_the_database_clock():
    from apps.knowledge_graph.services.builds import (
        _claim_locked_run,
        renew_build_lease,
        validate_build_lease,
    )

    for function in (validate_build_lease, _claim_locked_run, renew_build_lease):
        source = inspect.getsource(function)
        assert "Now()" in source
        assert "timezone.now()" not in source
    renewal_source = inspect.getsource(renew_build_lease)
    assert "lease_generation=lease_generation" in renewal_source
    assert "lease_expires_at__gt=Now()" in renewal_source


def test_lease_heartbeat_renews_immediately_and_surfaces_token_loss(monkeypatch):
    from apps.knowledge_graph.services import builds

    calls = []
    monkeypatch.setattr(
        builds,
        "renew_build_lease",
        lambda run_id, owner, generation: calls.append((run_id, owner, generation)),
    )
    with builds.LeaseHeartbeat(7, "owner", 3, interval_seconds=60) as heartbeat:
        heartbeat.pulse()
    assert calls == [(7, "owner", 3), (7, "owner", 3)]

    def lost(*_args):
        raise builds.BuildLeaseLostError("rotated")

    monkeypatch.setattr(builds, "renew_build_lease", lost)
    with pytest.raises(builds.BuildLeaseLostError, match="rotated"):
        with builds.LeaseHeartbeat(7, "owner", 3, interval_seconds=60):
            pass


def test_expensive_document_and_collection_stages_run_under_lease_heartbeat():
    from apps.knowledge_graph.services.builds import (
        build_document_graph,
        refresh_collection_graph,
    )

    document_source = inspect.getsource(build_document_graph)
    collection_source = inspect.getsource(refresh_collection_graph)
    assert document_source.count("LeaseHeartbeat(") >= 2
    assert collection_source.count("LeaseHeartbeat(") >= 2


def test_commit_markers_are_classified_as_absent_valid_or_corrupt():
    from apps.knowledge_graph.services.builds import (
        CommitMarkerState,
        _commit_marker_state,
    )

    assert (
        _commit_marker_state({}, "stage_commit", rows_present=False, valid=False)
        is CommitMarkerState.ABSENT
    )
    assert (
        _commit_marker_state({}, "stage_commit", rows_present=True, valid=False)
        is CommitMarkerState.CORRUPT
    )
    assert (
        _commit_marker_state(
            {"stage_commit": {}},
            "stage_commit",
            rows_present=False,
            valid=False,
        )
        is CommitMarkerState.CORRUPT
    )
    assert (
        _commit_marker_state(
            {"stage_commit": {"version": 1}},
            "stage_commit",
            rows_present=True,
            valid=True,
        )
        is CommitMarkerState.VALID
    )


def test_corrupt_commit_failure_is_permanent_for_the_exact_occurrence():
    from apps.knowledge_graph.services import builds

    with pytest.raises(builds.CorruptBuildError, match="permanently"):
        builds._validate_retryable_run(
            SimpleNamespace(error_code="corrupt_build_state")
        )
    builds._validate_retryable_run(SimpleNamespace(error_code="provider_unavailable"))
    assert "_validate_retryable_run(run)" in inspect.getsource(
        builds._bootstrap_document_build
    )
    assert "_validate_retryable_run(run)" in inspect.getsource(
        builds._bootstrap_collection_build
    )


def test_coordinators_fail_closed_on_stage_specific_commit_inspection():
    from apps.knowledge_graph.services import builds

    document_source = inspect.getsource(builds.build_document_graph)
    collection_source = inspect.getsource(builds.refresh_collection_graph)
    assert "_document_extraction_commit_state" in document_source
    assert "_document_resolution_commit_state" in document_source
    assert "_collection_resolution_commit_state" in collection_source
    assert "_collection_assembly_commit_state" in collection_source
    assert "_commit_marker_present" not in document_source
    assert "_commit_marker_present" not in collection_source
    assert "CommitMarkerState.CORRUPT" in document_source
    assert "CommitMarkerState.CORRUPT" in collection_source


def test_terminalization_uses_logical_scope_locks_without_requiring_sources():
    from apps.knowledge_graph.graph.assembly import (
        lock_collection_graph_advisory_scope,
        lock_collection_graph_scope,
    )
    from apps.knowledge_graph.services import builds

    document_source = inspect.getsource(builds._terminal_document_build)
    collection_source = inspect.getsource(builds._terminal_collection_build)
    assert "_lock_terminal_document_rows" in document_source
    assert "_lock_document_build_rows" not in document_source
    assert "_lock_terminal_collection_rows" in collection_source
    assert "_lock_collection_build_rows" not in collection_source
    assert "Collection.objects.select_for_update().filter" in inspect.getsource(
        builds._lock_terminal_collection_rows
    )
    assert "_lock_collection_scope" in inspect.getsource(lock_collection_graph_scope)
    assert "Collection.objects" not in inspect.getsource(
        lock_collection_graph_advisory_scope
    )


def test_terminal_bookkeeping_failure_never_masks_the_original_stage_error(monkeypatch):
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
    artifact = SimpleNamespace(pk=401)
    run = SimpleNamespace(
        pk=501,
        stage=GraphBuildRun.Stage.QUEUED,
        attempt=1,
        stats={},
    )
    monkeypatch.setattr(builds, "_document_context", lambda *_args: context)
    monkeypatch.setattr(
        builds,
        "_bootstrap_document_build",
        lambda *_args: (artifact, run, "owner", 3, False),
    )
    monkeypatch.setattr(builds, "_transition_run", _stage_transition_stub(run))
    monkeypatch.setattr(
        builds,
        "_document_extraction_commit_state",
        lambda *_args: builds.CommitMarkerState.ABSENT,
    )
    monkeypatch.setattr(
        builds, "LeaseHeartbeat", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(
        pipeline,
        "extract_into_build",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("original provider failure")
        ),
    )
    monkeypatch.setattr(
        builds,
        "_terminal_document_build",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("source row was deleted")
        ),
    )

    with pytest.raises(RuntimeError, match="original provider failure"):
        builds.build_document_graph(
            identity.document_id,
            identity.source_hash,
            builds.derive_document_build_key(identity),
        )


def test_collection_context_caps_documents_entities_chunks_and_characters_first():
    from apps.knowledge_graph.services import builds

    resolution_config = SimpleNamespace(max_document_inputs=2, max_entities=3)
    assembly_config = SimpleNamespace(
        max_document_inputs=2,
        max_entities=4,
        max_evidence=5,
    )
    builds._validate_collection_context_caps(
        document_count=2,
        entity_count=3,
        chunk_count=5,
        character_count=24,
        resolution_config=resolution_config,
        assembly_config=assembly_config,
        max_text_characters=8,
    )
    for changed, message in (
        ({"document_count": 3}, "document"),
        ({"entity_count": 4}, "entity"),
        ({"chunk_count": 6}, "chunk"),
        ({"character_count": 25}, "character"),
    ):
        values = {
            "document_count": 2,
            "entity_count": 3,
            "chunk_count": 5,
            "character_count": 24,
            **changed,
        }
        with pytest.raises(builds.CorruptBuildError, match=message):
            builds._validate_collection_context_caps(
                **values,
                resolution_config=resolution_config,
                assembly_config=assembly_config,
                max_text_characters=8,
            )

    source = inspect.getsource(builds._collection_context)
    assert "Document.filter(" not in source
    assert "_validate_collection_context_caps(" in source
    assert source.index("_validate_collection_context_caps(") < source.index(
        "current_chunks = _ordered_chunks("
    )
    assert source.index('Sum(Length("full_text"))') < source.index(
        "documents = tuple(document_values)"
    )
    assert "_bounded_context_rows(" in source
    assert "contributing_rows" not in source
    assert "collection awaits fresh document graph artifacts" in source


def test_collection_context_queryset_cap_bounds_actual_iteration_after_count_drift():
    from apps.knowledge_graph.services import builds

    bounded = getattr(builds, "_bounded_context_rows", None)
    assert callable(bounded)
    consumed = []

    class Query:
        def count(self):
            return 1

        def iterator(self, *, chunk_size):
            assert chunk_size == 2
            for value in range(10):
                consumed.append(value)
                yield value

    with pytest.raises(builds.CorruptBuildError, match="document.*cap"):
        bounded(Query(), 2, "collection document")
    assert consumed == [0, 1, 2]


def test_collection_return_active_revalidates_locked_manifest_before_success():
    from apps.knowledge_graph.services import builds

    bootstrap_source = inspect.getsource(builds._bootstrap_collection_build)
    revalidate = getattr(builds, "_revalidate_active_collection_build", None)
    assert callable(revalidate)
    revalidate_source = inspect.getsource(revalidate)
    return_active_branch = bootstrap_source[
        bootstrap_source.index(
            "if action is OccurrenceAction.RETURN_ACTIVE"
        ) : bootstrap_source.index("if action in {OccurrenceAction.RESUME")
    ]

    assert return_active_branch.index("_revalidate_active_collection_build(") < (
        return_active_branch.index("return artifact, run, None, None, True")
    )
    assert "validate_locked_active_collection_snapshot(" in revalidate_source
    assert "derive_collection_build_key(context.identity)" in revalidate_source
    assert "artifact.build_key != build_key" in revalidate_source
    assert "run.build_key != build_key" in revalidate_source
    assert "transaction.on_commit(" in bootstrap_source


def test_collection_return_active_classifies_locked_snapshot_corruption(monkeypatch):
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.services import builds

    identity = _collection_identity()
    build_key = builds.derive_collection_build_key(identity)
    context = SimpleNamespace(
        identity=identity,
        ontology=object(),
        assembly_config=object(),
    )
    artifact = SimpleNamespace(
        build_key=build_key,
        source_hash=identity.aggregate_source_signature,
        ontology_version=identity.ontology_version,
        ontology_checksum=identity.ontology_checksum,
        extractor_version=identity.extractor_version,
        resolver_version=identity.resolver_version,
        filter_policy_version=identity.filter_version,
        filter_policy_checksum=identity.filter_checksum,
        resolution_config_checksum=identity.resolver_checksum,
        assembly_version=identity.assembly_version,
        assembly_config_checksum=identity.assembly_checksum,
        embedding_model_signature=identity.embedding_model_signature,
    )
    run = SimpleNamespace(build_key=build_key)

    def reject_corrupt_snapshot(**_kwargs):
        raise assembly.CollectionGraphAssemblyError("corrupt marker")

    monkeypatch.setattr(
        assembly,
        "validate_locked_active_collection_snapshot",
        reject_corrupt_snapshot,
    )
    with pytest.raises(builds.CorruptBuildError, match="active collection snapshot"):
        builds._revalidate_active_collection_build(
            context,
            SimpleNamespace(pk=identity.collection_id),
            artifact,
            run,
            build_key,
        )


@pytest.mark.parametrize(
    ("target", "field_name"),
    (
        ("argument", None),
        ("artifact", "build_key"),
        ("run", "build_key"),
        ("artifact", "source_hash"),
        ("artifact", "ontology_version"),
        ("artifact", "ontology_checksum"),
        ("artifact", "extractor_version"),
        ("artifact", "resolver_version"),
        ("artifact", "filter_policy_version"),
        ("artifact", "filter_policy_checksum"),
        ("artifact", "resolution_config_checksum"),
        ("artifact", "assembly_version"),
        ("artifact", "assembly_config_checksum"),
        ("artifact", "embedding_model_signature"),
    ),
)
def test_collection_return_active_rejects_each_identity_mismatch(
    monkeypatch,
    target,
    field_name,
):
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.services import builds

    identity = _collection_identity()
    build_key = builds.derive_collection_build_key(identity)
    context = SimpleNamespace(
        identity=identity,
        ontology=object(),
        assembly_config=object(),
    )
    artifact = SimpleNamespace(
        build_key=build_key,
        source_hash=identity.aggregate_source_signature,
        ontology_version=identity.ontology_version,
        ontology_checksum=identity.ontology_checksum,
        extractor_version=identity.extractor_version,
        resolver_version=identity.resolver_version,
        filter_policy_version=identity.filter_version,
        filter_policy_checksum=identity.filter_checksum,
        resolution_config_checksum=identity.resolver_checksum,
        assembly_version=identity.assembly_version,
        assembly_config_checksum=identity.assembly_checksum,
        embedding_model_signature=identity.embedding_model_signature,
    )
    run = SimpleNamespace(build_key=build_key)
    validator_called = False

    def accept_snapshot(**_kwargs):
        nonlocal validator_called
        validator_called = True

    monkeypatch.setattr(
        assembly,
        "validate_locked_active_collection_snapshot",
        accept_snapshot,
    )
    requested_key = build_key
    if target == "argument":
        requested_key = "0" * 64
    else:
        setattr(artifact if target == "artifact" else run, field_name, object())

    with pytest.raises(builds.CorruptBuildError, match="requested build identity"):
        builds._revalidate_active_collection_build(
            context,
            SimpleNamespace(pk=identity.collection_id),
            artifact,
            run,
            requested_key,
        )
    assert validator_called is False


def test_tasks7_and_9_bound_occurrence_history_and_keep_monotonic_generation():
    from apps.knowledge_graph.extraction.pipeline import (
        _create_build_destination,
        _find_committed_extraction_run,
    )
    from apps.knowledge_graph.resolution.collection import (
        _lock_collection_destination_occurrence,
        persist_collection_resolution,
    )

    task7_create = inspect.getsource(_create_build_destination)
    assert "_lock_document_scope" in task7_create
    assert task7_create.index("_lock_document_scope(document_id)") < task7_create.index(
        "existing ="
    )
    assert "_find_committed_extraction_run(" in task7_create
    assert "for_update=True" in task7_create
    assert "legacy extraction identity is already in progress" in task7_create
    assert 'order_by("-build_generation", "-pk")' in task7_create
    assert "build_generation=build_generation" in task7_create

    task7_resume = inspect.getsource(_find_committed_extraction_run)
    assert "candidate_ids" in task7_resume
    assert "[:2]" in task7_resume
    assert "tuple(entity_query)" not in task7_resume
    assert "tuple(relation_query)" not in task7_resume

    task9_lock = inspect.getsource(_lock_collection_destination_occurrence)
    assert "candidate_artifact_id" in task9_lock
    assert "[:2]" in task9_lock
    task9_persist = inspect.getsource(persist_collection_resolution)
    assert "_lock_collection_destination_occurrence" in task9_persist
    assert "scope_artifacts = tuple(" not in task9_persist
    assert "scope_runs = tuple(" not in task9_persist


def test_document_resume_markers_and_resolver_cap_rows_before_materialization():
    from apps.knowledge_graph.services import builds

    extraction_source = inspect.getsource(builds._document_extraction_commit_state)
    resolution_source = inspect.getsource(builds._document_resolution_commit_state)
    coordinator_source = inspect.getsource(builds.build_document_graph)

    assert extraction_source.index("entity_count >") < extraction_source.index(
        "extraction_evidence_fingerprint("
    )
    assert extraction_source.index("relation_count >") < extraction_source.index(
        "extraction_evidence_fingerprint("
    )
    assert "tuple(entity_query)" not in extraction_source
    assert "tuple(relation_query)" not in extraction_source

    assert "tuple(mention_query)" not in resolution_source
    assert "tuple(entity_query.order_by" not in resolution_source
    assert "tuple(link_query.order_by" not in resolution_source
    assert resolution_source.count("_bounded_rows(") >= 3

    assert "_bounded_rows(" in coordinator_source
    assert coordinator_source.index("_bounded_rows(") < coordinator_source.index(
        "resolve_document_mentions("
    )


def test_recurrent_build_identity_creates_a_new_occurrence_after_an_intervening_key():
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.builds import (
        OccurrenceAction,
        _occurrence_action,
    )

    key_a = "a" * 64
    key_b = "b" * 64
    first_a = SimpleNamespace(
        pk=1,
        build_key=key_a,
        build_generation=1,
        status=GraphArtifact.Status.SUPERSEDED,
    )
    active_b = SimpleNamespace(
        pk=2,
        build_key=key_b,
        build_generation=2,
        status=GraphArtifact.Status.ACTIVE,
    )
    runs = (
        SimpleNamespace(
            artifact_id=1,
            build_key=key_a,
            build_generation=1,
            stage=GraphBuildRun.Stage.SUPERSEDED,
            status=GraphBuildRun.Status.CANCELLED,
        ),
        SimpleNamespace(
            artifact_id=2,
            build_key=key_b,
            build_generation=2,
            stage=GraphBuildRun.Stage.ACTIVE,
            status=GraphBuildRun.Status.SUCCEEDED,
        ),
    )

    assert (
        _occurrence_action((first_a, active_b), runs, key_a) is OccurrenceAction.CREATE
    )


def test_duplicate_delivery_joins_the_newest_current_occurrence():
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.builds import (
        OccurrenceAction,
        _occurrence_action,
    )

    key = "a" * 64
    historical = SimpleNamespace(
        pk=1,
        build_key=key,
        build_generation=1,
        status=GraphArtifact.Status.SUPERSEDED,
    )
    current = SimpleNamespace(
        pk=3,
        build_key=key,
        build_generation=3,
        status=GraphArtifact.Status.BUILDING,
    )
    runs = (
        SimpleNamespace(
            artifact_id=1,
            build_key=key,
            build_generation=1,
            stage=GraphBuildRun.Stage.SUPERSEDED,
            status=GraphBuildRun.Status.CANCELLED,
        ),
        SimpleNamespace(
            artifact_id=3,
            build_key=key,
            build_generation=3,
            stage=GraphBuildRun.Stage.RESOLVING,
            status=GraphBuildRun.Status.RUNNING,
        ),
    )

    assert (
        _occurrence_action((historical, current), runs, key) is OccurrenceAction.RESUME
    )


def test_active_exact_key_wins_even_if_a_newer_stale_occurrence_exists():
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.builds import (
        OccurrenceAction,
        _occurrence_action,
    )

    key = "a" * 64
    active = SimpleNamespace(
        pk=4,
        build_key=key,
        build_generation=4,
        status=GraphArtifact.Status.ACTIVE,
    )
    stale = SimpleNamespace(
        pk=5,
        build_key="b" * 64,
        build_generation=5,
        status=GraphArtifact.Status.STALE,
    )
    runs = (
        SimpleNamespace(
            artifact_id=4,
            build_key=key,
            build_generation=4,
            stage=GraphBuildRun.Stage.ACTIVE,
            status=GraphBuildRun.Status.SUCCEEDED,
        ),
        SimpleNamespace(
            artifact_id=5,
            build_key="b" * 64,
            build_generation=5,
            stage=GraphBuildRun.Stage.STALE,
            status=GraphBuildRun.Status.CANCELLED,
        ),
    )

    assert (
        _occurrence_action((active, stale), runs, key) is OccurrenceAction.RETURN_ACTIVE
    )


def test_next_build_generation_is_monotonic_for_a_scope_occurrence():
    from apps.knowledge_graph.services.builds import _next_build_generation

    artifacts = (
        SimpleNamespace(build_generation=1),
        SimpleNamespace(build_generation=7),
        SimpleNamespace(build_generation=3),
    )

    assert _next_build_generation(()) == 1
    assert _next_build_generation(artifacts) == 8


def test_scope_lockers_select_only_current_candidate_and_active_occurrences():
    from apps.knowledge_graph.graph.assembly import _locked_candidate
    from apps.knowledge_graph.services import builds

    document_source = inspect.getsource(builds._lock_document_build_rows)
    collection_source = inspect.getsource(builds._lock_collection_build_rows)
    assembly_source = inspect.getsource(_locked_candidate)

    assert "_bounded_scope_artifact_ids" in document_source
    assert "_bounded_scope_artifact_ids" in collection_source
    assert "pk__in=artifact_ids" in document_source
    assert "pk__in=artifact_ids" in collection_source
    assert "[:2]" in assembly_source
    assert ".filter(pk__in=artifact_ids)" in assembly_source


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


def test_task9_snapshot_accepts_the_serialized_artifact_occurrence():
    from apps.knowledge_graph.resolution.collection import build_collection_snapshot

    parameters = inspect.signature(build_collection_snapshot).parameters
    assert "build_generation" in parameters
    assert "orchestration_version" in parameters


def test_tasks7_to_10_propagate_typed_occurrence_identity_not_json_metadata():
    from apps.knowledge_graph.extraction.pipeline import (
        _mark_terminal,
        validate_build_identity,
    )
    from apps.knowledge_graph.graph.assembly import (
        _candidate_identity,
        _resolve_ontology,
        _swap_active_collection_artifact,
        _write_assembly,
    )
    from apps.knowledge_graph.resolution.collection import (
        _validate_collection_destination,
    )
    from apps.knowledge_graph.resolution.persistence import _validate_destination

    for function in (
        validate_build_identity,
        _validate_destination,
        _validate_collection_destination,
        _candidate_identity,
    ):
        source = inspect.getsource(function)
        assert '"build_generation"' in source
        assert '"orchestration_version"' in source

    for function in (
        _mark_terminal,
        _resolve_ontology,
        _write_assembly,
        _swap_active_collection_artifact,
    ):
        source = inspect.getsource(function)
        assert "orchestration_version" in source
        assert 'metadata.get("orchestration_version")' not in source


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
    robust_flags = []

    def capture_callback(callback, *, robust=False):
        callbacks.append(callback)
        robust_flags.append(robust)

    monkeypatch.setattr(builds.transaction, "on_commit", capture_callback)
    monkeypatch.setattr(
        builds,
        "_enqueue_current_collection_refresh",
        refreshed.append,
    )

    builds._register_document_refresh_callbacks(current, run)

    assert refreshed == []
    assert len(callbacks) == 2
    assert robust_flags == [True, True]
    for callback in callbacks:
        callback()
    assert refreshed == [17, 18]


def _orchestration_artifact(*, build_key: str = "7" * 64):
    from apps.knowledge_graph.models import GraphArtifact

    return GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        build_key=build_key,
        build_generation=1,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
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
        build_generation=artifact.build_generation,
        orchestration_version=artifact.orchestration_version,
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
        builds,
        "_document_extraction_commit_state",
        lambda *_args: builds.CommitMarkerState.ABSENT,
    )
    monkeypatch.setattr(
        builds, "LeaseHeartbeat", lambda *_args, **_kwargs: nullcontext()
    )
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
        builds,
        "_document_extraction_commit_state",
        lambda *_args: builds.CommitMarkerState.ABSENT,
    )
    monkeypatch.setattr(
        builds, "LeaseHeartbeat", lambda *_args, **_kwargs: nullcontext()
    )
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
        stats={"fixture_commit_state": "validated_by_inspector"},
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
        builds,
        "_document_extraction_commit_state",
        lambda *_args: builds.CommitMarkerState.VALID,
    )
    monkeypatch.setattr(
        builds,
        "_document_resolution_commit_state",
        lambda *_args: builds.CommitMarkerState.VALID,
    )
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
        stats={"fixture_commit_state": "validated_by_inspector"},
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
    assert calls == ["validate", "activate"]


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
        stats={"fixture_commit_state": "validated_by_inspector"},
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
        stats={"fixture_commit_state": "validated_by_inspector"},
    )
    terminal_calls = []
    monkeypatch.setattr(builds, "_collection_context", lambda *_args: context)
    monkeypatch.setattr(
        builds,
        "_bootstrap_collection_build",
        lambda *_args: (artifact, run, "owner", 3, False),
    )
    monkeypatch.setattr(
        builds,
        "_collection_assembly_commit_state",
        lambda *_args, **_kwargs: builds.CommitMarkerState.ABSENT,
    )
    monkeypatch.setattr(
        builds, "LeaseHeartbeat", lambda *_args, **_kwargs: nullcontext()
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
