from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


class _ArtifactQuery:
    def __init__(self, artifacts):
        self.artifacts = artifacts

    def using(self, _alias):
        return self

    def filter(self, **_kwargs):
        return self

    def order_by(self, *_fields):
        return self

    def iterator(self, **_kwargs):
        return iter(self.artifacts)


class _CollectionQuery:
    def __init__(self, existing_ids):
        self.existing_ids = existing_ids

    def using(self, _alias):
        return self

    def filter(self, **_kwargs):
        return self

    def values_list(self, *_args, **_kwargs):
        return self.existing_ids


def _migration_apps(scope_id, *, existing_ids):
    artifact_manager = _ArtifactQuery([SimpleNamespace(scope_id=scope_id)])
    artifact_manager.bulk_update = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("an invalid migration row must never be persisted")
    )
    graph_artifact = SimpleNamespace(objects=artifact_manager)
    collection = SimpleNamespace(objects=_CollectionQuery(existing_ids))

    def get_model(app_label, model_name):
        return {
            ("apps_knowledge_graph", "GraphArtifact"): graph_artifact,
            ("apps_collections", "Collection"): collection,
        }[(app_label, model_name)]

    return SimpleNamespace(get_model=get_model)


def test_collection_scope_backfill_rejects_signed_bigint_overflow():
    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0003_graph_lifecycle_ownership"
    )

    with pytest.raises(RuntimeError, match="signed bigint"):
        migration.backfill_collection_scopes(
            _migration_apps(str(2**63), existing_ids=()),
            SimpleNamespace(connection=SimpleNamespace(alias="default")),
        )


def test_collection_scope_backfill_fails_closed_for_an_unmapped_collection():
    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0003_graph_lifecycle_ownership"
    )

    with pytest.raises(RuntimeError, match="collection does not exist"):
        migration.backfill_collection_scopes(
            _migration_apps("17", existing_ids=()),
            SimpleNamespace(connection=SimpleNamespace(alias="default")),
        )
