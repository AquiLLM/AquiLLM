from __future__ import annotations

import dataclasses
import inspect
import os
import socket
import uuid
from types import SimpleNamespace

import pytest
from django.conf import settings

from apps.knowledge_graph.models import CanonicalEntity, CanonicalEntityLink


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


def test_permission_projection_filters_both_endpoints_before_crossing_identity():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalMembershipRow,
        PermissionBearingEntity,
        project_authorized_canonical_lookup,
    )

    endpoints = (
        PermissionBearingEntity(11, 1, 101, (uuid.UUID(int=1),)),
        PermissionBearingEntity(22, 2, 202, (uuid.UUID(int=2),)),
        PermissionBearingEntity(33, 3, 303, (uuid.UUID(int=3),)),
    )
    memberships = (
        CanonicalMembershipRow(900, 11, 1, 101),
        CanonicalMembershipRow(900, 22, 2, 202),
        CanonicalMembershipRow(900, 33, 3, 303),
    )

    result = project_authorized_canonical_lookup(
        seed_collection_entity_ids=(11,),
        permission_endpoints=endpoints,
        canonical_memberships=memberships,
        allowed_collection_ids=(1, 2),
        allowed_document_ids=(
            uuid.UUID(int=1),
            uuid.UUID(int=2),
        ),
        active_artifact_ids=(101, 202),
    )

    assert result.canonical_by_seed == ((11, 900),)
    assert result.members_by_canonical == ((900, ((1, 11, 101), (2, 22, 202))),)
    assert "33" not in repr(result)
    assert "303" not in repr(result)


def test_unauthorized_seed_cannot_discover_neighbor_existence_or_counts():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalMembershipRow,
        PermissionBearingEntity,
        project_authorized_canonical_lookup,
    )

    result = project_authorized_canonical_lookup(
        seed_collection_entity_ids=(99,),
        permission_endpoints=(
            PermissionBearingEntity(11, 1, 101, (uuid.UUID(int=1),)),
        ),
        canonical_memberships=(CanonicalMembershipRow(900, 11, 1, 101),),
        allowed_collection_ids=(1,),
        allowed_document_ids=(uuid.UUID(int=1),),
        active_artifact_ids=(101,),
    )

    assert result.canonical_by_seed == ()
    assert result.members_by_canonical == ()
    assert dataclasses.asdict(result) == {
        "canonical_by_seed": (),
        "members_by_canonical": (),
    }


def test_permission_lookup_contract_exposes_only_opaque_identifiers():
    from apps.knowledge_graph.resolution.canonical import CanonicalLookupResult

    fields = {field.name for field in dataclasses.fields(CanonicalLookupResult)}
    source = inspect.getsource(CanonicalLookupResult)

    assert fields == {"canonical_by_seed", "members_by_canonical"}
    assert "label" not in source
    assert "diagnostic" not in source
    assert "count" not in source


def test_db_lookup_requires_explicit_collection_and_document_allowlists():
    from apps.knowledge_graph.resolution.canonical import (
        authorized_canonical_lookup,
    )

    signature = inspect.signature(authorized_canonical_lookup)

    required = inspect.Parameter.empty
    assert signature.parameters["allowed_collection_ids"].default is required
    assert signature.parameters["allowed_document_ids"].default is required
    assert signature.parameters["active_artifact_ids"].default is required
    assert signature.parameters["resolver_version"].default is inspect.Parameter.empty

    with pytest.raises(ValueError, match="sorted"):
        authorized_canonical_lookup(
            seed_collection_entity_ids=(1,),
            allowed_collection_ids=(2, 1),
            allowed_document_ids=(uuid.UUID(int=1),),
            active_artifact_ids=(100, 200),
            resolver_version="canonical-resolution-v1",
        )


def test_current_link_boundary_filters_the_complete_live_path():
    source = inspect.getsource(type(CanonicalEntityLink.objects.all()).current)

    assert "artifact__status" in source
    assert "collection_entity__status" in source
    assert "canonical_entity__status" in source
    assert "outcome" in source
    assert "resolver_version" in source


def test_canonical_registry_is_not_a_permission_or_claim_container():
    forbidden = {
        "collection",
        "document",
        "chunk",
        "claim",
        "relation",
        "user",
        "group",
        "permission",
    }
    concrete_fields = {field.name for field in CanonicalEntity._meta.fields}

    assert forbidden.isdisjoint(concrete_fields)


def test_v1_lookup_is_db_derived_and_has_no_shared_cache_authority():
    from apps.knowledge_graph.resolution import canonical

    source = inspect.getsource(canonical.authorized_canonical_lookup)

    assert "django.core.cache" not in inspect.getsource(canonical)
    assert ".current(" in source
    assert "allowed_collection_ids" in source
    assert "allowed_document_ids" in source


def test_direct_collection_activation_schedules_reconciliation_after_commit():
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.resolution import canonical

    swap_source = inspect.getsource(assembly._swap_active_collection_artifact)
    schedule_source = inspect.getsource(canonical.schedule_canonical_rebuild)

    assert "schedule_canonical_rebuild" in swap_source
    assert "transaction.on_commit" in schedule_source
    assert "robust=True" in schedule_source


def test_rebuild_scheduler_dedupes_only_callbacks_still_queued(monkeypatch):
    from django.db import transaction

    from apps.knowledge_graph.resolution import canonical

    connection = SimpleNamespace(run_on_commit=[])

    def capture(callback, *, using, robust):
        connection.run_on_commit.append((frozenset({"inner"}), callback, robust))

    monkeypatch.setattr(transaction, "get_connection", lambda _using: connection)
    monkeypatch.setattr(transaction, "on_commit", capture)

    canonical.schedule_canonical_rebuild()
    canonical.schedule_canonical_rebuild()
    assert len(connection.run_on_commit) == 1
    assert connection.run_on_commit[0][2] is True

    # Django discards an inner savepoint's callbacks on rollback.  A later
    # successful activation in the same outer transaction must queue again.
    connection.run_on_commit.clear()
    canonical.schedule_canonical_rebuild()
    assert len(connection.run_on_commit) == 1


def test_real_activation_schedules_once_and_idempotent_reactivation_schedules_zero(
    monkeypatch,
):
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.resolution import canonical

    scheduled = []
    monkeypatch.setattr(
        canonical, "schedule_canonical_rebuild", lambda: scheduled.append("rebuild")
    )
    artifact = SimpleNamespace(
        pk=1,
        build_generation=1,
        status=GraphArtifact.Status.BUILDING,
        activated_at=None,
        completed_at=None,
        superseded_at=None,
        save=lambda **_kwargs: None,
    )
    run = SimpleNamespace(
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        stage=GraphBuildRun.Stage.VALIDATING,
        status=GraphBuildRun.Status.RUNNING,
        finished_at=None,
        lease_owner="worker",
        lease_expires_at=object(),
        save=lambda **_kwargs: None,
    )

    assembly._swap_active_collection_artifact(
        artifact=artifact,
        run=run,
        scope_artifacts=(artifact,),
    )
    assert scheduled == ["rebuild"]

    assembly._swap_active_collection_artifact(
        artifact=artifact,
        run=run,
        scope_artifacts=(artifact,),
    )
    assert scheduled == ["rebuild"]


def _building_collection_artifact(collection, digit):
    from apps.knowledge_graph.models import GraphArtifact

    return GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=collection.pk,
        status=GraphArtifact.Status.BUILDING,
        source_hash=digit * 64,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="collection-resolution-v1",
        filter_policy_version="filter-v1",
        embedding_model_signature=(
            f"test-local:model@rev:endpoint={'e' * 64}:dims=1024:"
            "prep=kg-entity-v1:max_chars=8192:batch=64"
        ),
    )


@pytest.mark.django_db(transaction=True)
@database_required
def test_activation_on_commit_first_replace_rollback_idempotent_and_robust(
    monkeypatch,
):
    from django.db import transaction

    from apps.collections.models import Collection
    from apps.knowledge_graph.graph import assembly
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.resolution import canonical

    calls = []
    monkeypatch.setattr(
        canonical,
        "rebuild_canonical_registry",
        lambda **_kwargs: calls.append("rebuilt"),
    )
    collection = Collection.objects.create(name="canonical activation")
    first = _building_collection_artifact(collection, "1")
    first_run = GraphBuildRun.objects.create(
        artifact=first,
        stage=GraphBuildRun.Stage.PERSISTENCE,
        status=GraphBuildRun.Status.RUNNING,
    )

    with transaction.atomic():
        assembly._swap_active_collection_artifact(
            artifact=first,
            run=first_run,
            scope_artifacts=(first,),
        )
        assert calls == []
    assert calls == ["rebuilt"]

    with transaction.atomic():
        assembly._swap_active_collection_artifact(
            artifact=first,
            run=first_run,
            scope_artifacts=(first,),
        )
    assert calls == ["rebuilt"]

    replacement = _building_collection_artifact(collection, "2")
    replacement_run = GraphBuildRun.objects.create(
        artifact=replacement,
        stage=GraphBuildRun.Stage.PERSISTENCE,
        status=GraphBuildRun.Status.RUNNING,
    )
    with transaction.atomic():
        assembly._swap_active_collection_artifact(
            artifact=replacement,
            run=replacement_run,
            scope_artifacts=(first, replacement),
        )
    assert calls == ["rebuilt", "rebuilt"]

    rollback_collection = Collection.objects.create(name="canonical rollback")
    rolled_back = _building_collection_artifact(rollback_collection, "3")
    rollback_run = GraphBuildRun.objects.create(
        artifact=rolled_back,
        stage=GraphBuildRun.Stage.PERSISTENCE,
        status=GraphBuildRun.Status.RUNNING,
    )
    with pytest.raises(RuntimeError, match="outer rollback"):
        with transaction.atomic():
            assembly._swap_active_collection_artifact(
                artifact=rolled_back,
                run=rollback_run,
                scope_artifacts=(rolled_back,),
            )
            raise RuntimeError("outer rollback")
    rolled_back.refresh_from_db()
    assert rolled_back.status == GraphArtifact.Status.BUILDING
    assert calls == ["rebuilt", "rebuilt"]

    robust_collection = Collection.objects.create(name="canonical robust")
    robust_artifact = _building_collection_artifact(robust_collection, "4")
    robust_run = GraphBuildRun.objects.create(
        artifact=robust_artifact,
        stage=GraphBuildRun.Stage.PERSISTENCE,
        status=GraphBuildRun.Status.RUNNING,
    )

    def fail_rebuild(**_kwargs):
        calls.append("failed")
        raise RuntimeError("injected canonical rebuild failure")

    monkeypatch.setattr(canonical, "rebuild_canonical_registry", fail_rebuild)
    with transaction.atomic():
        assembly._swap_active_collection_artifact(
            artifact=robust_artifact,
            run=robust_run,
            scope_artifacts=(robust_artifact,),
        )
    robust_artifact.refresh_from_db()
    assert robust_artifact.status == GraphArtifact.Status.ACTIVE
    assert calls[-1] == "failed"
