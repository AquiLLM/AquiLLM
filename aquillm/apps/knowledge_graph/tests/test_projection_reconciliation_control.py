from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from uuid import uuid4

from apps.knowledge_graph.projection import generation_audit, reconciler
from apps.knowledge_graph.projection.records import (
    ProjectionCountsV1,
    ProjectionLifecycleState,
)


def _settings():
    return SimpleNamespace(
        projection_batch_size=25,
        graph_overall_timeout_ms=500,
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
        relation_count=2,
        evidence_count=3,
        chunk_count=4,
        lease_expires_at=None,
    )


def _bundle(generation_key="d" * 64):
    counts = ProjectionCountsV1(1, 1, 1, 4, 2, 3, 1)
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


def test_reconcile_replays_only_missing_expired_or_drifted_work(monkeypatch):
    rows = {
        11: SimpleNamespace(id=uuid4(), state="ready"),
        22: SimpleNamespace(id=uuid4(), state="building"),
        33: SimpleNamespace(id=uuid4(), state="ready"),
    }
    reasons = {11: None, 22: "expired_lease", 33: "checksum_drift"}
    pages = [((1, 11), (2, 22), (3, 33)), ()]
    monkeypatch.setattr(
        reconciler, "_active_artifact_page", lambda **_kwargs: pages.pop(0)
    )
    monkeypatch.setattr(
        reconciler,
        "_projection_for_active",
        lambda **kwargs: rows[kwargs["artifact_id"]],
    )
    monkeypatch.setattr(
        reconciler,
        "_generation_audit",
        lambda **kwargs: SimpleNamespace(
            replay_reason=reasons[
                next(key for key, value in rows.items() if value is kwargs["row"])
            ]
        ),
    )
    monkeypatch.setattr(reconciler, "_orphan_generation_keys", lambda **_kwargs: ())
    monkeypatch.setattr(reconciler, "_projection_settings", _settings)
    monkeypatch.setattr(
        reconciler, "projection_identifier_codec", lambda _value: object()
    )
    monkeypatch.setattr(reconciler, "_postgres_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_memgraph_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_atomic", lambda _using: nullcontext())
    enqueued = []
    superseded = []
    monkeypatch.setattr(
        reconciler,
        "supersede_projection_locked",
        lambda **kwargs: superseded.append(kwargs["projection_id"]),
    )
    monkeypatch.setattr(
        reconciler,
        "enqueue_collection_projection_locked",
        lambda **kwargs: enqueued.append(kwargs["artifact_id"]),
    )

    summary = reconciler.reconcile_graph_projections(page_size=3, dry_run=False)

    assert enqueued == [22, 33]
    assert superseded == [rows[33].id]
    assert summary.enqueued_count == 2
    assert summary.drift_count == 1
    assert summary.replayed_count == 2


def test_global_orphan_scan_uses_exclusive_opaque_cursor(monkeypatch):
    authoritative = _bundle("1" * 64)
    row = SimpleNamespace(id=1, state="ready")
    projection_pages = [(row,), ()]
    monkeypatch.setattr(
        generation_audit,
        "_projection_page",
        lambda **_kwargs: projection_pages.pop(0),
    )
    postgres = SimpleNamespace(load_projection_bundle=lambda **_kwargs: authoritative)
    observed_cursors = []

    def list_generations(**kwargs):
        observed_cursors.append(kwargs["after_generation_key"])
        if kwargs["after_generation_key"] is None:
            return (_manifest(authoritative), _manifest(_bundle("2" * 64)))
        return ()

    graph = SimpleNamespace(list_generations=list_generations)
    settings = _settings()
    settings.projection_batch_size = 2

    orphaned = generation_audit.orphan_generation_keys(
        postgres=postgres,
        graph=graph,
        settings=settings,
        limit=10,
    )

    assert tuple(key.value for key in orphaned) == ("2" * 64,)
    assert observed_cursors[0] is None
    assert observed_cursors[1].value == "2" * 64


def test_collection_orphan_scan_never_reads_other_collection_manifests(monkeypatch):
    authoritative = _bundle("1" * 64)
    projection_pages = [(SimpleNamespace(id=1, state="ready"),), ()]
    monkeypatch.setattr(
        generation_audit,
        "_projection_page",
        lambda **kwargs: (
            projection_pages.pop(0) if kwargs["collection_id"] == 7 else ()
        ),
    )
    observed = []
    graph = SimpleNamespace(
        list_generations=lambda **kwargs: observed.append(kwargs) or ()
    )

    generation_audit.orphan_generation_keys(
        postgres=SimpleNamespace(
            load_projection_bundle=lambda **_kwargs: authoritative
        ),
        graph=graph,
        settings=_settings(),
        limit=10,
        collection_id=7,
        collection_key=generation_audit._opaque_generation("e" * 64),
    )

    assert observed[0]["collection_key"].value == "e" * 64


def test_authoritative_scan_loads_only_graph_eligible_lifecycle_purposes(monkeypatch):
    rows = (
        SimpleNamespace(id=1, state="pending"),
        SimpleNamespace(id=2, state="building"),
        SimpleNamespace(id=3, state="ready"),
        SimpleNamespace(id=4, state="superseded"),
        SimpleNamespace(
            id=5,
            state="superseded",
            collection_id=None,
            artifact_id=None,
        ),
    )
    pages = [rows, ()]
    monkeypatch.setattr(
        generation_audit,
        "_projection_page",
        lambda **_kwargs: pages.pop(0),
    )
    observed = []

    def load(**kwargs):
        observed.append((kwargs["projection_id"], kwargs["purpose"]))
        return _bundle(str(kwargs["projection_id"]) * 64)

    keys = generation_audit._authoritative_generation_keys(
        postgres=SimpleNamespace(load_projection_bundle=load),
        settings=_settings(),
        collection_id=None,
    )

    assert observed == [(2, "build"), (3, "audit"), (4, "prune")]
    assert keys == frozenset(("2" * 64, "3" * 64, "4" * 64))
