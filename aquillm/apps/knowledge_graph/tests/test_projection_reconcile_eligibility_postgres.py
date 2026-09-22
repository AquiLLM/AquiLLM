"""Reconciliation must skip stale collection authority without hiding failures."""

import os
from types import SimpleNamespace

import pytest
from django.db import connection
from django.utils import timezone

from apps.collections.models import Collection
from apps.knowledge_graph.models import (
    CollectionGraphMembershipState,
    CollectionGraphProjection,
)
from apps.knowledge_graph.projection import reconciler
from apps.knowledge_graph.projection.state_repository import (
    FunctionProjectionStateRepository,
)
from apps.knowledge_graph.tests.test_projection_locking_postgres import (
    _active_artifact,
)

pytestmark = [
    pytest.mark.django_db(transaction=True, databases="__all__"),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="requires disposable PostgreSQL",
    ),
]


def _authority(name, *, membership=True):
    assert connection.vendor == "postgresql"
    collection = Collection.objects.create(name=name)
    artifact = _active_artifact(collection)
    if membership:
        CollectionGraphMembershipState.objects.create(
            collection=collection,
            active_artifact=artifact,
            registry_epoch=1,
            membership_checksum="a" * 64,
            resolver_version=artifact.resolver_version,
            resolution_config_checksum=artifact.resolution_config_checksum,
        )
    return collection, artifact


def test_active_page_requires_membership_for_exact_collection_and_artifact():
    missing, missing_artifact = _authority("missing", membership=False)
    current, current_artifact = _authority("current")
    mismatch, mismatch_artifact = _authority("mismatch", membership=False)
    CollectionGraphMembershipState.objects.create(
        collection=mismatch,
        active_artifact=missing_artifact,
        registry_epoch=1,
        membership_checksum="b" * 64,
        resolver_version="resolver-v1",
        resolution_config_checksum=missing_artifact.resolution_config_checksum,
    )

    assert reconciler._active_artifact_page(after_id=0, page_size=10) == (
        (current.pk, current_artifact.pk),
    )
    assert reconciler._active_artifact_page(
        after_id=0, page_size=1, collection_id=missing.pk
    ) == ()
    assert reconciler._active_artifact_page(
        after_id=current_artifact.pk, page_size=10
    ) == ()
    assert mismatch_artifact.pk > current_artifact.pk


def test_empty_replay_result_has_specific_stale_authority_exception():
    collection, artifact = _authority("stale replay", membership=False)
    with pytest.raises(RuntimeError) as caught:
        FunctionProjectionStateRepository().replay(
            projection_id=None,
            collection_id=collection.pk,
            artifact_id=artifact.pk,
            versions=("collection-graph-v1", "projection-v1", "key-v1"),
            now=timezone.now(),
        )
    assert type(caught.value).__name__ == "StaleProjectionAuthority"
    assert not CollectionGraphProjection.objects.exists()


@pytest.mark.parametrize("previous_ready", [False, True])
def test_authority_changed_after_page_does_not_abort_following_collection(
    monkeypatch, previous_ready
):
    stale, stale_artifact = _authority("stale during audit")
    current, current_artifact = _authority("still current")
    prior = None
    if previous_ready:
        prior = CollectionGraphProjection.objects.create(
            collection=stale,
            collection_pk_snapshot=stale.pk,
            artifact=stale_artifact,
            artifact_pk_snapshot=stale_artifact.pk,
            state="ready",
            schema_version="collection-graph-v1",
            projection_version="projection-v1",
            identifier_key_version="key-v1",
            membership_epoch=1,
            membership_checksum="a" * 64,
            private_mapping_checksum="b" * 64,
            graph_checksum="c" * 64,
            snapshot_checksum="d" * 64,
            ready_at=timezone.now(),
        )
    settings = SimpleNamespace(
        projection_schema_version="collection-graph-v1",
        projection_format_version="projection-v1",
        projection_identifier_key_version="key-v1",
    )
    monkeypatch.setattr(reconciler, "_projection_settings", lambda: settings)
    monkeypatch.setattr(reconciler, "_postgres_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_memgraph_repository", lambda: object())
    monkeypatch.setattr(reconciler, "projection_identifier_codec", lambda _: object())
    monkeypatch.setattr(reconciler, "_orphan_generation_keys", lambda **_: ())

    def audit(**_kwargs):
        # Deterministic race: page already selected both artifacts, then authority
        # disappears before the real state function locks and checks membership.
        CollectionGraphMembershipState.objects.filter(collection=stale).delete()
        return SimpleNamespace(replay_reason="missing_authority")

    monkeypatch.setattr(reconciler, "_generation_audit", audit)
    summary = reconciler.reconcile_graph_projections(page_size=2, dry_run=False)

    assert summary.examined_count == 2
    assert summary.enqueued_count == 1
    assert list(
        CollectionGraphProjection.objects.filter(state="pending").values_list(
            "collection_id", "artifact_id"
        )
    ) == [(current.pk, current_artifact.pk)]
    if prior is not None:
        prior.refresh_from_db()
        assert prior.state == "ready"
        assert prior.superseded_at is None
    else:
        assert not CollectionGraphProjection.objects.filter(
            artifact=stale_artifact
        ).exists()
