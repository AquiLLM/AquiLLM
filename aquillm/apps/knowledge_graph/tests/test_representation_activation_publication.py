"""Durable representation rebuild publication is strictly after outer commit."""

from uuid import uuid4

import pytest
from django.db import connection, transaction

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.knowledge_graph.models import GraphRebuildRequest
from apps.knowledge_graph.projection.representation_lifecycle import (
    prepare_document_representation_activation,
)
from apps.knowledge_graph.services import builds
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture

pytestmark = pytest.mark.django_db(transaction=True)


def test_nondefault_publication_alias_is_closed_before_mutation(monkeypatch):
    fixture = ScopeFixture(monkeypatch)
    projection = fixture.persist_ready("A")
    ref = fixture.snapshot.selections[0].document
    with transaction.atomic(), pytest.raises(SourceScopeUnavailable):
        prepare_document_representation_activation(
            ref,
            (7,),
            expected_source_hash=ref.source_hash,
            old_representation_key=None,
            new_representation_key=uuid4(),
            using="projection_source",
        )
    projection.refresh_from_db()
    assert projection.state == "ready"
    assert not GraphRebuildRequest.objects.exists()


@pytest.mark.parametrize("action", ["commit", "rollback", "publish_failure"])
def test_activation_publication_follows_outer_commit(monkeypatch, action):
    fixture = ScopeFixture(monkeypatch)
    projection = fixture.persist_ready("A")
    ref = fixture.snapshot.selections[0].document
    calls = []

    def resume(request_id):
        assert not connection.in_atomic_block
        assert GraphRebuildRequest.objects.filter(pk=request_id).exists()
        calls.append(request_id)

    def fail_publish(*args, **kwargs):
        assert not connection.in_atomic_block
        calls.append(kwargs["request_id"])
        raise RuntimeError("synthetic broker unavailable")

    if action == "publish_failure":
        monkeypatch.setattr(builds, "enqueue_document_build", fail_publish)
    else:
        monkeypatch.setattr(builds, "resume_rebuild_request", resume)
    try:
        with transaction.atomic():
            prepare_document_representation_activation(
                ref,
                (7,),
                expected_source_hash=ref.source_hash,
                old_representation_key=None,
                new_representation_key=uuid4(),
                using="default",
            )
            assert calls == []
            assert GraphRebuildRequest.objects.count() == 1
            if action == "rollback":
                raise RuntimeError("rollback")
    except RuntimeError:
        assert action == "rollback"
    projection.refresh_from_db()
    if action == "rollback":
        assert calls == [] and not GraphRebuildRequest.objects.exists()
        assert projection.state == "ready"
    else:
        request = GraphRebuildRequest.objects.get()
        assert calls == [request.pk]
        assert projection.state == "superseded"
        assert request.document_publication_state == (
            "failed" if action == "publish_failure" else "pending"
        )
