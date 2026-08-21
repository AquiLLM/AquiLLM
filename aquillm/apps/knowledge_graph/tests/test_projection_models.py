from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from django.core.exceptions import ValidationError
from django.db import models

from apps.knowledge_graph.models.projections import (
    CollectionGraphMembershipState,
    CollectionGraphProjection,
    GraphProjectionOutbox,
    ProjectionChunkReference,
)

HEX = "a" * 64


def _projection(**overrides: object) -> CollectionGraphProjection:
    values: dict[str, object] = {
        "collection_pk_snapshot": 7,
        "collection_id": 7,
        "artifact_pk_snapshot": 11,
        "artifact_id": 11,
        "state": CollectionGraphProjection.State.PENDING,
        "schema_version": "schema-v1",
        "projection_version": "projection-v1",
        "identifier_key_version": "key-v1",
        "membership_epoch": 0,
        "membership_checksum": HEX,
        "private_mapping_checksum": HEX,
    }
    values.update(overrides)
    return CollectionGraphProjection(**values)


def test_membership_state_contract_is_bounded_and_tombstone_safe() -> None:
    collection = CollectionGraphMembershipState._meta.get_field("collection")
    artifact = CollectionGraphMembershipState._meta.get_field("active_artifact")

    assert collection.one_to_one and collection.primary_key
    assert collection.remote_field.on_delete is models.CASCADE
    assert artifact.remote_field.on_delete is models.SET_NULL
    assert artifact.null

    state = CollectionGraphMembershipState(
        collection_id=7,
        registry_epoch=2**63,
        membership_checksum=HEX,
        resolver_version="resolver-v1",
        resolution_config_checksum=HEX,
    )
    with pytest.raises(ValidationError, match="registry_epoch"):
        state.clean()


@pytest.mark.parametrize("field", ["membership_checksum", "resolution_config_checksum"])
def test_membership_state_rejects_noncanonical_checksums(field: str) -> None:
    state = CollectionGraphMembershipState(
        collection_id=7,
        membership_checksum=HEX,
        resolver_version="resolver-v1",
        resolution_config_checksum=HEX,
    )
    setattr(state, field, "A" * 64)

    with pytest.raises(ValidationError, match=field):
        state.clean()


def test_projection_lifecycle_requires_exact_terminal_metadata() -> None:
    ready = _projection(
        state=CollectionGraphProjection.State.READY,
        graph_checksum=HEX,
        snapshot_checksum=HEX,
        ready_at=datetime.now(UTC),
        lease_owner="worker-secret",
    )
    with pytest.raises(ValidationError, match="lease"):
        ready.clean()

    failed = _projection(state=CollectionGraphProjection.State.FAILED)
    with pytest.raises(ValidationError, match="failure_code"):
        failed.clean()

    superseded = _projection(state=CollectionGraphProjection.State.SUPERSEDED)
    with pytest.raises(ValidationError, match="superseded_at"):
        superseded.clean()


@pytest.mark.parametrize(
    "count_field",
    ("entity_count", "relation_semantics_count", "entity_mention_count"),
)
def test_projection_rejects_invalid_versions_counts_and_checksums(
    count_field: str,
) -> None:
    projection = _projection(schema_version=" bad", **{count_field: -1})

    with pytest.raises(ValidationError) as captured:
        projection.clean()

    assert {"schema_version", count_field} <= set(captured.value.message_dict)


def test_projection_fields_and_constraints_preserve_authoritative_tombstones() -> None:
    collection = CollectionGraphProjection._meta.get_field("collection")
    artifact = CollectionGraphProjection._meta.get_field("artifact")
    generation = CollectionGraphProjection._meta.get_field("generation_key")
    names = {
        constraint.name for constraint in CollectionGraphProjection._meta.constraints
    }

    assert collection.null and collection.remote_field.on_delete is models.SET_NULL
    assert artifact.null and artifact.remote_field.on_delete is models.SET_NULL
    assert generation.unique and generation.editable is False
    assert {
        "kg_projection_generation_unique",
        "kg_projection_active_identity_unique",
        "kg_projection_nonnegative_counts",
        "kg_projection_lease_pair",
        "kg_projection_lifecycle_valid",
    } <= names


def test_chunk_reference_and_outbox_have_exact_identity_constraints() -> None:
    chunk = ProjectionChunkReference._meta.get_field("chunk")
    reference_names = {
        constraint.name for constraint in ProjectionChunkReference._meta.constraints
    }
    outbox_names = {
        constraint.name for constraint in GraphProjectionOutbox._meta.constraints
    }

    assert chunk.null and chunk.remote_field.on_delete is models.SET_NULL
    assert {
        "kg_projection_chunk_key_unique",
        "kg_projection_chunk_coordinate_unique",
    } <= reference_names
    assert {
        "kg_projection_outbox_operation_unique",
        "kg_projection_outbox_state_valid",
    } <= outbox_names


def test_projection_primary_and_generation_keys_are_independent_uuids() -> None:
    projection = _projection()

    assert isinstance(projection.id, uuid.UUID)
    assert isinstance(projection.generation_key, uuid.UUID)
    assert projection.id != projection.generation_key


@pytest.mark.parametrize(
    "state",
    [
        CollectionGraphProjection.State.PENDING,
        CollectionGraphProjection.State.BUILDING,
        CollectionGraphProjection.State.READY,
    ],
)
def test_active_projection_states_require_live_collection_and_artifact(
    state: str,
) -> None:
    values: dict[str, object] = {"state": state, "collection_id": None}
    if state == CollectionGraphProjection.State.BUILDING:
        values.update(lease_owner="worker", lease_expires_at=datetime.now(UTC))
    if state == CollectionGraphProjection.State.READY:
        values.update(
            graph_checksum=HEX,
            snapshot_checksum=HEX,
            ready_at=datetime.now(UTC),
        )
    with pytest.raises(ValidationError, match="collection"):
        _projection(**values).clean()


def test_terminal_projection_tombstones_may_retain_snapshots_with_null_fks() -> None:
    failed = _projection(
        state=CollectionGraphProjection.State.FAILED,
        collection_id=None,
        artifact_id=None,
        failure_code="source_changed",
    )
    superseded = _projection(
        state=CollectionGraphProjection.State.SUPERSEDED,
        collection_id=None,
        artifact_id=None,
        superseded_at=datetime.now(UTC),
    )

    failed.clean()
    superseded.clean()
    assert (failed.collection_pk_snapshot, failed.artifact_pk_snapshot) == (7, 11)
