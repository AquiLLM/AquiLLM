from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from apps.knowledge_graph.projection import reconciler
from apps.knowledge_graph.projection.records import ProjectionCountsV1


def _settings():
    return SimpleNamespace(
        projection_batch_size=25,
        graph_overall_timeout_ms=500,
    )


def _bundle(generation_key="d" * 64):
    return SimpleNamespace(
        generation=SimpleNamespace(generation_key=generation_key),
        counts=ProjectionCountsV1(1, 1, 1, 4, 2, 3, 1),
    )


def test_prune_honors_projection_id_and_uses_persisted_opaque_key(monkeypatch):
    row = SimpleNamespace(id=uuid4())
    observed = {}
    monkeypatch.setattr(
        reconciler,
        "_prune_candidates",
        lambda **kwargs: observed.update(filters=kwargs) or (row,),
    )
    monkeypatch.setattr(reconciler, "_orphan_generation_keys", lambda **_kwargs: ())
    monkeypatch.setattr(
        reconciler,
        "_postgres_repository",
        lambda: SimpleNamespace(load_projection_bundle=lambda **_kwargs: _bundle()),
    )
    deleted = []
    monkeypatch.setattr(
        reconciler,
        "_memgraph_repository",
        lambda: SimpleNamespace(
            delete_generation=lambda **kwargs: deleted.append(
                kwargs["generation_key"].value
            )
        ),
    )
    monkeypatch.setattr(reconciler, "_projection_settings", _settings)

    summary = reconciler.prune_graph_projection_generations(
        projection_id=row.id,
        collection_id=None,
        page_size=10,
        retain=2,
        dry_run=False,
    )

    assert observed["filters"]["projection_id"] == row.id
    assert deleted == ["d" * 64]
    assert summary.deleted_count == 1


def test_exact_projection_prune_is_not_hidden_by_collection_retention(monkeypatch):
    projection_id = uuid4()
    row = SimpleNamespace(id=projection_id)
    filters = []

    class Query:
        def filter(self, **kwargs):
            filters.append(kwargs)
            return self

        def annotate(self, **_kwargs):
            raise AssertionError("exact projection pruning must not rank retention")

        def order_by(self, *_args):
            return self

        def __getitem__(self, _value):
            return (row,)

    monkeypatch.setattr(
        reconciler.CollectionGraphProjection,
        "objects",
        SimpleNamespace(filter=lambda **kwargs: Query().filter(**kwargs)),
    )

    candidates = reconciler._prune_candidates(
        page_size=10,
        retain=2,
        projection_id=projection_id,
        collection_id=None,
    )

    assert candidates == (row,)
    assert {"pk": projection_id} in filters


def test_inspection_reports_manifest_drift_and_orphans_not_failed_alias(monkeypatch):
    class Query:
        def all(self):
            return self

        def order_by(self, *_args):
            return self

        def values_list(self, *_args, **_kwargs):
            return ("ready", "failed")

    monkeypatch.setattr(reconciler.CollectionGraphProjection, "objects", Query())
    monkeypatch.setattr(
        reconciler,
        "reconcile_graph_projections",
        lambda **_kwargs: reconciler.ReconcileSummaryV1(
            2, 0, True, drift_count=0, orphan_count=3, replayed_count=1
        ),
    )

    counts = reconciler.inspect_projection_authority(
        collection_id=None,
        all_collections=True,
        page_size=10,
    )

    assert counts["failed_count"] == 1
    assert counts["drift_count"] == 0
    assert counts["orphan_count"] == 3
    assert counts["replayed_count"] == 1
