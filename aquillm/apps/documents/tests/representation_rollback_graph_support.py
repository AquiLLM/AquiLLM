"""Persisted synthetic graph lifecycle rows bound to the actual current source.

These rows exercise PostgreSQL invalidation, not graph extraction or remote readiness.
"""
from django.utils import timezone

from apps.knowledge_graph.models import (
    CollectionArtifactInput, CollectionGraphProjection, GraphArtifact,
    GraphBuildRun, GraphProjectionOutbox,
)
from apps.knowledge_graph.projection.representation_inputs import binding_values
from apps.knowledge_graph.tests.test_models import _artifact
from .representation_rollback_support import frozen


def seed_graph(doc):
    snapshot = frozen(doc)
    binding = binding_values(snapshot)
    assert snapshot.selections[0].generation_key is not None
    document_artifact = _artifact(scope_id=doc.id, source_hash=doc.full_text_hash,
        status="active", **binding)
    document_artifact.save()
    collection_artifact = _artifact(scope_type="collection", scope_id=doc.collection_id,
        status="building", **binding)
    collection_artifact.save()
    manifest = CollectionArtifactInput.objects.create(artifact=collection_artifact,
        collection_id=doc.collection_id, document_id=doc.id,
        document_artifact=document_artifact)
    collection_artifact.status = "active"
    collection_artifact.save(update_fields=["status"])
    now = timezone.now()
    run = GraphBuildRun.objects.create(artifact=collection_artifact,
        stage=GraphBuildRun.Stage.COMPLETE, status=GraphBuildRun.Status.SUCCEEDED,
        finished_at=now)
    projection = CollectionGraphProjection.objects.create(
        collection_id=doc.collection_id, collection_pk_snapshot=doc.collection_id,
        artifact=collection_artifact, artifact_pk_snapshot=collection_artifact.pk,
        state="ready", ready_at=now, schema_version="collection-graph-v1",
        projection_version="projection-v1", identifier_key_version="key-v1",
        membership_epoch=0, membership_checksum="1" * 64,
        private_mapping_checksum="2" * 64, graph_checksum="3" * 64,
        snapshot_checksum="3" * 64, **binding)
    rows = dict(document_artifact=document_artifact, collection_artifact=collection_artifact,
        manifest=manifest, run=run, projection=projection)
    before = capture_graph(rows)
    assert before["prune"] == []
    return rows, before


def capture_graph(rows):
    captured = {name: type(row).objects.filter(pk=row.pk).values().get()
                for name, row in rows.items()}
    captured["prune"] = list(GraphProjectionOutbox.objects.filter(
        projection=rows["projection"]).order_by("pk").values())
    return captured


def assert_graph_superseded(rows, before):
    after = capture_graph(rows)
    for name in ("document_artifact", "collection_artifact"):
        assert after[name]["status"] == GraphArtifact.Status.SUPERSEDED
        assert after[name]["superseded_at"] is not None
        changed = {"status", "superseded_at", "updated_at"}
        assert {k: v for k, v in after[name].items() if k not in changed} == {
            k: v for k, v in before[name].items() if k not in changed}
    assert after["manifest"] == before["manifest"]
    projection = after["projection"]
    assert projection["state"] == "superseded" and projection["superseded_at"] is not None
    assert projection["ready_at"] == before["projection"]["ready_at"]
    assert projection["generation_key"] == before["projection"]["generation_key"]
    assert projection["representation_digest"] == before["projection"]["representation_digest"]
    assert projection["lease_owner"] == "" and projection["lease_expires_at"] is None
    assert len(after["prune"]) == 1
    assert after["prune"][0]["operation"] == "prune" and after["prune"][0]["state"] == "pending"
    run = after["run"]
    assert (run["stage"], run["status"], run["error_code"]) == (
        "stale", "cancelled", "representation_changed")
    assert run["finished_at"] == before["run"]["finished_at"]
    assert run["artifact_id"] == before["run"]["artifact_id"]
    assert run["lease_owner"] == "" and run["lease_expires_at"] is None
