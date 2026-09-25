from __future__ import annotations

import inspect
import uuid

import pytest


def test_extraction_persistence_takes_document_advisory_before_artifact_rows():
    from apps.knowledge_graph.extraction.pipeline import (
        _lock_extraction_orchestration_rows,
    )

    source = inspect.getsource(_lock_extraction_orchestration_rows)
    advisory = source.index("lock_document_graph_advisory_scope(document_id)")
    artifact = source.index(
        "GraphArtifact.objects.select_for_update().get(",
        advisory,
    )

    assert advisory < artifact


def test_extraction_lock_helper_executes_advisory_before_row_locks(monkeypatch):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services import builds

    actions = []

    class _Manager:
        def __init__(self, label):
            self.label = label

        def select_for_update(self):
            actions.append(f"{self.label}_select_for_update")
            return self

        def get(self, **kwargs):
            actions.append((f"{self.label}_get", kwargs))
            return self.label

    monkeypatch.setattr(
        builds,
        "lock_document_graph_advisory_scope",
        lambda document_id: actions.append(("document_advisory", document_id)),
    )
    monkeypatch.setattr(GraphArtifact, "objects", _Manager("artifact"))
    monkeypatch.setattr(GraphBuildRun, "objects", _Manager("run"))
    document_id = __import__("uuid").uuid4()

    result = pipeline._lock_extraction_orchestration_rows(11, 13, document_id)

    assert result == ("artifact", "run")
    assert actions == [
        ("document_advisory", document_id),
        "artifact_select_for_update",
        ("artifact_get", {"pk": 11}),
        "run_select_for_update",
        ("run_get", {"pk": 13}),
    ]


@pytest.mark.parametrize(
    "loader_name",
    ("load_collection_resolution_inputs", "load_collection_filter_inputs"),
)
def test_task9_loaders_take_collection_prefix_and_lock_manifest_rows_only(loader_name):
    from apps.knowledge_graph.resolution import collection

    source = inspect.getsource(getattr(collection, loader_name))
    collection_lock = source.index("lock_collection_graph_scope(collection_id)")
    artifact_lock = source.index("GraphArtifact.objects.select_for_update().get(")

    assert collection_lock < artifact_lock
    assert "lock_collection_manifest_sources(" in source
    assert "CollectionArtifactInput.objects.select_for_update" not in source


def test_shared_manifest_protocol_locks_complete_document_scope_before_manifest_rows():
    from apps.knowledge_graph.graph.manifest_locking import (
        lock_collection_manifest_sources,
    )

    source = inspect.getsource(lock_collection_manifest_sources)
    advisory = source.index("lock_document_graph_advisory_scope(document_id)")
    artifact = source.index("_lock_graph_artifacts(")
    run = source.index("_lock_graph_runs(")
    document = source.index("_lock_exact_document_rows(")
    manifest = source.index("locked_rows = _read_manifest(")

    assert advisory < artifact < run < document < manifest

    from apps.knowledge_graph.graph.manifest_locking import _read_manifest

    assert 'select_for_update(of=("self",))' in inspect.getsource(_read_manifest)


def test_manifest_document_advisories_use_the_origin_scope_key_order():
    from apps.knowledge_graph.graph.manifest_locking import _ordered_document_ids
    from apps.knowledge_graph.services.builds import document_graph_advisory_lock_key

    document_ids = tuple(uuid.UUID(int=value, version=4) for value in (91, 2, 57, 11))

    assert _ordered_document_ids(document_ids) == tuple(
        sorted(
            document_ids,
            key=lambda value: (document_graph_advisory_lock_key(value), value.int),
        )
    )


def test_manifest_identity_revalidation_rejects_late_expansion():
    from apps.knowledge_graph.graph.manifest_locking import (
        ManifestLockError,
        _assert_manifest_identity_unchanged,
    )

    before = ((1, 2, 3, uuid.UUID(int=4), 5, "a", "b", "c"),)
    after = (*before, (9, 2, 3, uuid.UUID(int=8), 7, "a", "b", "c"))

    with pytest.raises(ManifestLockError, match="changed"):
        _assert_manifest_identity_unchanged(before, after)


def test_manifest_concrete_row_locks_batch_by_model_and_predicate_cap(monkeypatch):
    from types import SimpleNamespace

    from django.apps import apps as django_apps

    from apps.knowledge_graph.graph import manifest_locking

    calls = []
    rows_by_label = {
        "apps_documents.rawtextdocument": tuple(
            SimpleNamespace(pkid=value, id=uuid.UUID(int=value, version=4))
            for value in range(1, 6)
        ),
        "apps_documents.pdfdocument": tuple(
            SimpleNamespace(pkid=value, id=uuid.UUID(int=100 + value, version=4))
            for value in range(1, 3)
        ),
    }

    class Query:
        def __init__(self, label, rows):
            self.label = label
            self.rows = rows

        def select_for_update(self):
            calls.append(self.label)
            return self

        def filter(self, **kwargs):
            pkids = set(kwargs["pkid__in"])
            return Query(self.label, tuple(row for row in self.rows if row.pkid in pkids))

        def order_by(self, *_fields):
            return tuple(sorted(self.rows, key=lambda row: (row.pkid, row.id.int)))

    models = {
        label: SimpleNamespace(_base_manager=Query(label, rows))
        for label, rows in rows_by_label.items()
    }
    monkeypatch.setattr(django_apps, "get_model", models.__getitem__)
    monkeypatch.setattr(manifest_locking, "_QUERY_BATCH_SIZE", 2)
    refs = tuple(
        manifest_locking._DocumentRowRef(label, row.pkid, row.id)
        for label, rows in rows_by_label.items()
        for row in reversed(rows)
    )

    locked = manifest_locking._lock_exact_document_rows(refs)

    assert len(locked) == 7
    assert calls.count("apps_documents.rawtextdocument") == 3
    assert calls.count("apps_documents.pdfdocument") == 1


def test_all_manifest_consumers_use_the_shared_document_first_protocol():
    from apps.knowledge_graph.graph import assembly, filtering
    from apps.knowledge_graph.resolution import collection

    consumers = (
        collection.load_collection_resolution_inputs,
        collection.load_collection_filter_inputs,
        collection.persist_collection_resolution,
        assembly._load_filter_source_lineage,
        assembly._load_locked_manifest,
        filtering._validate_existing_filter_rerun,
        filtering.create_filter_rerun_artifact,
    )
    for consumer in consumers:
        source = inspect.getsource(consumer)
        assert "lock_collection_manifest_sources(" in source, consumer.__name__
        assert (
            "CollectionArtifactInput.objects.select_for_update" not in source
        ), consumer.__name__


def test_manifest_snapshot_reuses_prelocked_sources_and_concrete_documents():
    from apps.knowledge_graph.resolution.collection import (
        _snapshot_from_locked_manifest,
    )

    source = inspect.getsource(_snapshot_from_locked_manifest)

    assert "isinstance(manifest_rows, LockedCollectionManifest)" in source
    assert "manifest_rows.document_artifacts" in source
    assert "manifest_rows.documents" in source


def test_collection_contributor_validation_resolves_logical_document_uuid():
    from apps.knowledge_graph.graph.assembly import _lock_current_contributors

    source = inspect.getsource(_lock_current_contributors)

    assert ".get(id=source.scope_id)" in source
    assert ".get(pk=source.scope_id)" not in source
