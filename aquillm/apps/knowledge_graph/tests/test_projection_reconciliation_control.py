# ruff: noqa: F401
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

from django.utils import timezone

from apps.knowledge_graph.projection import generation_audit, reconciler
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.projection.records import (
    ProjectionCountsV1,
    ProjectionLifecycleState,
)


def _settings():
    return SimpleNamespace(
        projection_batch_size=25,
        graph_overall_timeout_ms=500,
        projection_schema_version="collection-graph-v1",
        projection_format_version="projection-v1",
        projection_identifier_key_version="key-v7",
    )


def _ready_row():
    return SimpleNamespace(
        id=uuid4(),
        state="ready",
        schema_version="collection-graph-v1",
        projection_version="projection-v1",
        identifier_key_version="key-v7",
        membership_epoch=3,
        membership_checksum="a" * 64,
        graph_checksum="b" * 64,
        snapshot_checksum="b" * 64,
        private_mapping_checksum="c" * 64,
        entity_count=1,
        relation_semantics_count=1,
        relation_count=2,
        evidence_count=3,
        entity_mention_count=2,
        chunk_count=4,
        lease_expires_at=None,
    )


def _bundle(generation_key="d" * 64):
    counts = ProjectionCountsV1(1, 1, 1, 4, 1, 2, 3, 2, 1)
    return SimpleNamespace(
        generation=SimpleNamespace(
            generation_key=generation_key,
            collection_key="e" * 64,
            schema_version="collection-graph-v1",
            projection_version="projection-v1",
            identifier_key_version="key-v7",
            membership_epoch=3,
            membership_checksum="a" * 64,
        ),
        counts=counts,
    )


def _manifest(bundle):
    return SimpleNamespace(
        generation_key=bundle.generation.generation_key,
        schema_version=bundle.generation.schema_version,
        projection_version=bundle.generation.projection_version,
        identifier_key_version=bundle.generation.identifier_key_version,
        graph_checksum="b" * 64,
        snapshot_checksum="b" * 64,
        private_mapping_checksum="c" * 64,
        counts=bundle.counts,
        state=ProjectionLifecycleState.READY,
    )


def test_generation_audit_detects_empty_store_and_checksum_drift(monkeypatch):
    row = _ready_row()
    bundle = _bundle()
    purposes = []
    postgres = SimpleNamespace(
        load_projection_bundle=lambda **kwargs: (
            purposes.append(kwargs["purpose"]) or bundle
        )
    )
    graph = SimpleNamespace(
        read_generation_manifest=lambda **_kwargs: _manifest(bundle),
        validate_generation=lambda **_kwargs: SimpleNamespace(valid=True),
    )
    monkeypatch.setattr(
        generation_audit,
        "projection_checksum",
        lambda _bundle: "b" * 64,
    )

    healthy = generation_audit.audit_projection_generation(
        row=row,
        postgres=postgres,
        graph=graph,
        settings=_settings(),
    )
    graph.read_generation_manifest = lambda **_kwargs: (_ for _ in ()).throw(
        ValueError("generation marker is missing")
    )
    missing = generation_audit.audit_projection_generation(
        row=row,
        postgres=postgres,
        graph=graph,
        settings=_settings(),
    )
    graph.read_generation_manifest = lambda **_kwargs: SimpleNamespace(
        **{**vars(_manifest(bundle)), "graph_checksum": "f" * 64}
    )
    drift = generation_audit.audit_projection_generation(
        row=row,
        postgres=postgres,
        graph=graph,
        settings=_settings(),
    )

    assert healthy.replay_reason is None
    assert missing.replay_reason == "missing_generation"
    assert drift.replay_reason == "checksum_drift"
    assert purposes == ["audit", "audit", "audit"]


def test_generation_audit_leaves_inflight_projection_for_queued_worker() -> None:
    pending_row = _ready_row()
    pending_row.state = "pending"
    building_row = _ready_row()
    building_row.state = "building"
    building_row.lease_expires_at = timezone.now() + timedelta(seconds=30)
    pending = generation_audit.audit_projection_generation(
        row=pending_row,
        postgres=object(),
        graph=object(),
        settings=_settings(),
    )
    building = generation_audit.audit_projection_generation(
        row=building_row,
        postgres=object(),
        graph=object(),
        settings=_settings(),
    )

    assert pending.replay_reason is None
    assert building.replay_reason is None


def test_ready_marker_does_not_hide_mutated_or_deleted_graph_records(monkeypatch):
    row = _ready_row()
    bundle = _bundle()
    validations = []

    def validate(**kwargs):
        validations.append(kwargs["expected"])
        return SimpleNamespace(valid=False)

    graph = SimpleNamespace(
        read_generation_manifest=lambda **_kwargs: _manifest(bundle),
        validate_generation=validate,
    )
    monkeypatch.setattr(
        generation_audit, "projection_checksum", lambda _value: "b" * 64
    )

    audit = generation_audit.audit_projection_generation(
        row=row,
        postgres=SimpleNamespace(load_projection_bundle=lambda **_kwargs: bundle),
        graph=graph,
        settings=_settings(),
    )

    assert audit.replay_reason == "checksum_drift"
    assert validations[0].private_mapping_checksum == row.private_mapping_checksum
    assert validations[0].state is ProjectionLifecycleState.READY
