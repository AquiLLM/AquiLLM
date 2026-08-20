from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from uuid import uuid4

from apps.knowledge_graph.projection import reconciler


def test_project_generation_replays_partial_staging_and_ready_cas(monkeypatch):
    projection_id = uuid4()
    calls = []
    lease = SimpleNamespace(projection_id=str(projection_id))
    bundle = SimpleNamespace(generation=SimpleNamespace(generation_key="a" * 64))
    validation = SimpleNamespace(valid=True, validation_checksum="b" * 64)
    source = SimpleNamespace(load_projection_bundle=lambda **_kwargs: bundle)
    graph = SimpleNamespace(
        write_staging_generation=lambda **_kwargs: calls.append("stage"),
        validate_generation=lambda **_kwargs: validation,
        mark_generation_ready=lambda **_kwargs: calls.append("graph_ready"),
    )
    monkeypatch.setattr(reconciler, "claim_projection_lease", lambda **_kwargs: lease)
    monkeypatch.setattr(reconciler, "_postgres_repository", lambda: source)
    monkeypatch.setattr(reconciler, "_memgraph_repository", lambda: graph)
    monkeypatch.setattr(reconciler, "_expected_manifest", lambda *_args: object())
    monkeypatch.setattr(
        reconciler,
        "publish_projection_ready_compare_and_set",
        lambda **_kwargs: SimpleNamespace(published=True, failure_code=None),
    )

    outcome = reconciler.project_generation(
        projection_id=projection_id, lease_owner="worker-a"
    )

    assert outcome.ready is True
    assert calls == ["stage", "graph_ready"]


def test_project_generation_redacts_partial_write_failures(monkeypatch):
    projection_id = uuid4()
    monkeypatch.setattr(
        reconciler,
        "claim_projection_lease",
        lambda **_kwargs: SimpleNamespace(projection_id=str(projection_id)),
    )
    monkeypatch.setattr(
        reconciler,
        "_postgres_repository",
        lambda: SimpleNamespace(
            load_projection_bundle=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("secret payload")
            )
        ),
    )
    failed = []
    monkeypatch.setattr(
        reconciler,
        "mark_projection_failed",
        lambda **kwargs: failed.append(kwargs["failure_code"]),
    )

    outcome = reconciler.project_generation(
        projection_id=projection_id, lease_owner="worker-a"
    )

    assert outcome.ready is False and outcome.failure_code == "write_failed"
    assert "secret" not in repr(outcome)
    assert len(failed) == 1


def test_reconcile_handles_empty_store_drift_and_newer_artifact_in_pages(monkeypatch):
    pages = [((1, 11), (2, 22)), ((3, 33),), ()]
    monkeypatch.setattr(
        reconciler, "_active_artifact_page", lambda **_kwargs: pages.pop(0)
    )
    monkeypatch.setattr(reconciler, "_atomic", lambda _using: nullcontext())
    enqueued = []
    monkeypatch.setattr(
        reconciler,
        "enqueue_collection_projection_locked",
        lambda **kwargs: enqueued.append(
            (kwargs["collection_id"], kwargs["artifact_id"])
        ),
    )

    summary = reconciler.reconcile_graph_projections(page_size=2, dry_run=False)

    assert summary.examined_count == 3
    assert enqueued == [(1, 11), (2, 22), (3, 33)]


def test_pruning_is_bounded_and_dry_run_never_deletes(monkeypatch):
    rows = tuple(SimpleNamespace(id=uuid4()) for _ in range(3))
    monkeypatch.setattr(reconciler, "_prune_candidates", lambda **_kwargs: rows)
    deleted = []
    monkeypatch.setattr(
        reconciler, "_delete_projection_generation", lambda row: deleted.append(row.id)
    )

    dry = reconciler.prune_graph_projection_generations(
        page_size=3, retain=1, dry_run=True
    )

    assert dry.candidate_count == 3 and deleted == []
