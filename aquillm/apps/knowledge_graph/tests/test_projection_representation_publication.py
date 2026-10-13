"""The real projection CAS refuses a changed captured representation source."""

from datetime import timedelta

import pytest
from django.utils import timezone

from apps.documents.models import RawTextDocument
from apps.knowledge_graph.models import GraphProjectionOutbox
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture
from apps.knowledge_graph.tests.test_canonical_reconciliation_deadline_postgres import (
    other_connection_publication,
)

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("changed", [False, True])
def test_captured_source_compare_and_set(monkeypatch, changed):
    fixture = ScopeFixture(monkeypatch)
    projection = fixture.persist_ready("A", captured=True)
    projection.state, projection.ready_at = "building", None
    projection.lease_owner = "g5"
    projection.lease_expires_at = timezone.now() + timedelta(seconds=30)
    projection.save()
    monkeypatch.setattr(
        "apps.knowledge_graph.projection.lifecycle._projection_versions",
        lambda: ("collection-graph-v1", "projection-v1", "key-v1"),
    )
    if changed:
        RawTextDocument.objects.filter(pk=fixture.documents["A"].pk).update(
            full_text="replacement source"
        )
    _, published = other_connection_publication(projection.pk)
    assert published is not changed
    projection.refresh_from_db()
    assert projection.state == ("superseded" if changed else "ready")
    assert (
        GraphProjectionOutbox.objects.filter(
            projection=projection, operation="prune", state="pending"
        ).exists()
        is changed
    )
