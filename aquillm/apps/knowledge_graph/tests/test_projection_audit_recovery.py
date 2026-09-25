from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection import generation_audit, reconciler, tasks


@pytest.mark.parametrize("present", [True, False])
def test_prune_records_completion_so_later_pages_can_progress(monkeypatch, present):
    rows = [SimpleNamespace(id=uuid4(), generation_key=uuid4()) for _ in range(3)]
    completed, deleted = set(), []
    monkeypatch.setattr(reconciler, "_projection_settings", lambda: object())
    monkeypatch.setattr(reconciler, "_memgraph_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_orphan_generation_keys", lambda **kwargs: ())
    monkeypatch.setattr(reconciler, "_postgres_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_prepare_prune", lambda row: True, raising=False)
    monkeypatch.setattr(
        reconciler,
        "_prune_candidates",
        lambda **kwargs: tuple(row for row in rows if row.id not in completed)[
            : kwargs["page_size"]
        ],
    )
    monkeypatch.setattr(
        reconciler, "_record_pruned", lambda row: completed.add(row.id), raising=False
    )
    monkeypatch.setattr(
        reconciler,
        "_delete_projection_generation",
        lambda **kwargs: deleted.append(kwargs["row"].id) or present,
    )

    for _ in range(3):
        reconciler.prune_graph_projection_generations(
            page_size=1, retain=1, dry_run=False
        )

    assert deleted == [row.id for row in rows]
    assert completed == {row.id for row in rows}


def test_prune_does_not_delete_a_failed_generation_reclaimed_by_worker(monkeypatch):
    row = SimpleNamespace(id=uuid4(), generation_key=uuid4(), state="failed")
    monkeypatch.setattr(reconciler, "_projection_settings", lambda: object())
    monkeypatch.setattr(reconciler, "_memgraph_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_prune_candidates", lambda **kwargs: (row,))
    monkeypatch.setattr(reconciler, "_prepare_prune", lambda row: False, raising=False)
    deleted = []
    monkeypatch.setattr(
        reconciler,
        "_delete_projection_generation",
        lambda **kwargs: deleted.append(row.id),
    )
    monkeypatch.setattr(reconciler, "_record_pruned", lambda row: None)
    reconciler.prune_graph_projection_generations(
        projection_id=row.id, page_size=1, retain=1, dry_run=False
    )
    assert deleted == []


def test_full_recovery_page_schedules_followup(monkeypatch):
    deliveries, scheduled = [], []
    counts = iter((0, 2))

    def publish(**kwargs):
        count = next(counts)
        deliveries.append(count)
        return SimpleNamespace(
            attempted_count=count, published_count=count, failed_count=0
        )

    monkeypatch.setattr(tasks, "publish_projection_outbox", publish)
    monkeypatch.setattr(
        tasks,
        "reconcile_graph_projections",
        lambda **kwargs: SimpleNamespace(examined_count=0, enqueued_count=2),
    )
    monkeypatch.setattr(
        tasks.reconcile_knowledge_graph_projections,
        "apply_async",
        lambda **kwargs: scheduled.append(kwargs),
    )
    result = tasks.reconcile_knowledge_graph_projections.run(2, False, 7)
    assert result["published_count"] == 2
    assert scheduled == [
        {"kwargs": {"page_size": 2, "collection_id": 7}, "countdown": 1}
    ]


def test_reconcile_publishes_work_created_during_its_run(monkeypatch):
    pending, delivered = [], []

    def publish(**kwargs):
        count = len(pending)
        delivered.extend(pending)
        pending.clear()
        return SimpleNamespace(
            attempted_count=count,
            published_count=count,
            failed_count=0,
        )

    def reconcile(**kwargs):
        pending.append("recovered-generation")
        return SimpleNamespace(examined_count=1, enqueued_count=1)

    monkeypatch.setattr(tasks, "publish_projection_outbox", publish)
    monkeypatch.setattr(tasks, "reconcile_graph_projections", reconcile)
    result = tasks.reconcile_knowledge_graph_projections.run(10, False, 7)
    assert delivered == ["recovered-generation"]
    assert pending == []
    assert result["published_count"] == 1


@pytest.mark.parametrize(
    "changed",
    ["schema_version", "projection_version", "identifier_key_version"],
)
def test_old_ready_version_is_replayed_without_decoding_it(changed):
    versions = {
        "schema_version": "schema-v1",
        "projection_version": "projection-v1",
        "identifier_key_version": "key-v1",
    }
    row = SimpleNamespace(id=uuid4(), state="ready", **versions)
    configured = dict(versions)
    configured[changed] = "version-v2"
    settings = SimpleNamespace(
        projection_schema_version=configured["schema_version"],
        projection_format_version=configured["projection_version"],
        projection_identifier_key_version=configured["identifier_key_version"],
        projection_batch_size=10,
    )

    def incompatible_loader(**kwargs):
        raise ValueError("old bundle must not be decoded with new versions")

    result = generation_audit.audit_projection_generation(
        row=row,
        postgres=SimpleNamespace(load_projection_bundle=incompatible_loader),
        graph=object(),
        settings=settings,
    )
    assert result.replay_reason == "version_changed"
