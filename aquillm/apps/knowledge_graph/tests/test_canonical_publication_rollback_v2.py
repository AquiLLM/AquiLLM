"""A nonempty previous registry survives stale and expired V2 publication."""

from time import monotonic

import pytest
from django.db.models import QuerySet

from apps.collections.models import Collection
from apps.knowledge_graph.models import CanonicalEntity, CanonicalEntityLink
from apps.knowledge_graph.resolution import canonical
from apps.knowledge_graph.resolution.canonical_reconciliation_v2 import (
    reconcile_canonical_v2,
)
from apps.knowledge_graph.tests.test_canonical_resolution import (
    _create_active_collection_entity,
)

pytestmark = pytest.mark.django_db(transaction=True)


def registry_rows():
    return (
        tuple(CanonicalEntity.objects.order_by("pk").values()),
        tuple(CanonicalEntityLink.objects.order_by("pk").values()),
    )


@pytest.mark.parametrize("race", ["stale", "expired_after_write"])
def test_previous_registry_survives_publication_failure(monkeypatch, race):
    artifact, entity = _create_active_collection_entity(
        Collection.objects.create(name="Retained registry"),
        label="Aquila",
        cluster_digit="a",
        identifier="repository:github.com/example/aquila",
    )
    _create_active_collection_entity(
        Collection.objects.create(name="Second registry member"),
        label="Aquila",
        cluster_digit="b",
        identifier="repository:github.com/example/aquila",
    )
    canonical.rebuild_canonical_registry(using="default")
    before = registry_rows()
    assert before[0] and before[1]
    elapsed = [0.0]
    calls = []
    if race == "stale":
        real = canonical.resolve_canonical_entities

        def compute(*args, **kwargs):
            calls.append(1)
            artifact.status = "superseded"
            artifact.save(update_fields=["status"])
            return real(*args, **kwargs)

        monkeypatch.setattr(canonical, "resolve_canonical_entities", compute)
    else:
        real = canonical._reconcile_locked_registry

        def publish(*args, **kwargs):
            calls.append(1)
            result = real(*args, **kwargs)
            # A deliberate write proves the caller transaction rolls back, too.
            QuerySet.update(CanonicalEntity.objects.all(), label="must roll back")
            elapsed[0] = 5.0
            return result

        monkeypatch.setattr(canonical, "_reconcile_locked_registry", publish)
    start = monotonic()
    outcome = reconcile_canonical_v2(
        using="default",
        deadline=start + 2,
        clock=lambda: monotonic() + elapsed[0],
    )
    assert outcome.status == ("stale" if race == "stale" else "timed_out")
    assert calls == [1]
    assert registry_rows() == before
