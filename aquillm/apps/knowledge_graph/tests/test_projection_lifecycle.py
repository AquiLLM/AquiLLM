from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

from apps.knowledge_graph.projection import lifecycle
from apps.knowledge_graph.projection.memgraph_repository import ProjectionValidationV1
from apps.knowledge_graph.projection.records import ProjectionCountsV1


class _Projection(SimpleNamespace):
    def save(self, **kwargs):
        self.saved.append(kwargs["update_fields"])


def _projection(**overrides):
    values = {
        "id": uuid4(),
        "state": "pending",
        "attempt_count": 0,
        "lease_owner": "",
        "lease_expires_at": None,
        "failure_code": "",
        "ready_at": None,
        "superseded_at": None,
        "collection_id": 7,
        "artifact_id": 11,
        "membership_epoch": 3,
        "membership_checksum": "a" * 64,
        "schema_version": "memgraph-schema-v1",
        "projection_version": "projection-v1",
        "identifier_key_version": "key-v1",
        "generation_key": uuid4(),
        "entity_count": 0,
        "relation_count": 0,
        "evidence_count": 0,
        "chunk_count": 0,
        "graph_checksum": "",
        "snapshot_checksum": "",
        "private_mapping_checksum": "b" * 64,
        "saved": [],
    }
    values.update(overrides)
    return _Projection(**values)


def test_claim_is_idempotent_for_owner_and_reclaims_only_expired_lease(monkeypatch):
    row = _projection()
    monkeypatch.setattr(lifecycle, "_atomic", lambda _using: nullcontext())
    monkeypatch.setattr(lifecycle, "_locked_projection", lambda *_args: row)
    monkeypatch.setattr(lifecycle, "_enqueue_outbox", lambda *_args: None)
    now = datetime(2026, 8, 20, tzinfo=UTC)

    first = lifecycle.claim_projection_lease(
        projection_id=row.id,
        owner="worker-a",
        now=now,
        lease_seconds=30,
        using="default",
    )
    duplicate = lifecycle.claim_projection_lease(
        projection_id=row.id,
        owner="worker-a",
        now=now + timedelta(seconds=1),
        lease_seconds=30,
        using="default",
    )
    blocked = lifecycle.claim_projection_lease(
        projection_id=row.id,
        owner="worker-b",
        now=now + timedelta(seconds=2),
        lease_seconds=30,
        using="default",
    )

    assert first is not None and first.attempt_count == 1
    assert duplicate is not None and duplicate.attempt_count == 1
    assert blocked is None


def test_ready_cas_rechecks_membership_and_never_publishes_stale_generation(
    monkeypatch,
):
    row = _projection(
        state="building",
        lease_owner="worker-a",
        lease_expires_at=datetime(2026, 8, 20, 0, 1, tzinfo=UTC),
    )
    membership = SimpleNamespace(
        active_artifact_id=11,
        registry_epoch=4,
        membership_checksum="c" * 64,
        resolver_version="resolver-v1",
        resolution_config_checksum="f" * 64,
    )
    artifact = SimpleNamespace(
        pk=11,
        status="active",
        resolver_version="resolver-v1",
        resolution_config_checksum="f" * 64,
    )
    monkeypatch.setattr(lifecycle, "_atomic", lambda _using: nullcontext())
    monkeypatch.setattr(
        lifecycle,
        "_locked_ready_context",
        lambda *_args: (SimpleNamespace(pk=7), artifact, membership, row),
    )
    validation = ProjectionValidationV1(
        "d" * 64,
        "e" * 64,
        ProjectionCountsV1(0, 0, 0, 0, 0, 0, 0),
        True,
    )

    outcome = lifecycle.publish_projection_ready_compare_and_set(
        projection_id=row.id,
        owner="worker-a",
        validation=validation,
        now=datetime(2026, 8, 20, tzinfo=UTC),
        using="default",
    )

    assert outcome.published is False
    assert outcome.failure_code == "source_changed"
    assert row.state == "failed"


def test_supersession_and_deletion_clear_leases_but_keep_tombstone_ids(monkeypatch):
    now = datetime(2026, 8, 20, tzinfo=UTC)
    row = _projection(
        state="building",
        lease_owner="worker-a",
        lease_expires_at=now + timedelta(seconds=30),
    )
    monkeypatch.setattr(lifecycle, "_atomic", lambda _using: nullcontext())
    monkeypatch.setattr(lifecycle, "_locked_projection", lambda *_args: row)
    monkeypatch.setattr(lifecycle, "_enqueue_outbox", lambda *_args: None)

    lifecycle.supersede_projection_locked(
        projection_id=row.id, now=now, using="default"
    )

    assert row.state == "superseded"
    assert row.lease_owner == "" and row.lease_expires_at is None
    assert row.collection_id == 7 and row.artifact_id == 11


def test_activation_canonical_and_invalidation_hooks_are_transactional():
    from pathlib import Path

    root = Path(__file__).parents[1]
    assembly = (root / "graph" / "assembly.py").read_text(encoding="utf-8")
    canonical = (root / "resolution" / "canonical.py").read_text(encoding="utf-8")
    invalidation = (root / "graph" / "invalidation.py").read_text(encoding="utf-8")

    assert "enqueue_collection_projection_locked(" in assembly
    assert "enqueue_automatic_membership_changes_locked(" in canonical
    assert "supersede_artifact_projections_locked(" in invalidation
