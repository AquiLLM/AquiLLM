"""The additive provenance columns cannot be erased after their first use."""

import pytest
from django.db import IntegrityError, connection, transaction
from django.db.migrations.exceptions import IrreversibleError
from django.db.migrations.executor import MigrationExecutor
from django.db.models import QuerySet

from apps.documents.services.representation_contracts import RepresentationSnapshotV1
from apps.knowledge_graph.models import GraphArtifact
from apps.knowledge_graph.projection.representation_inputs import binding_values
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture
from apps.knowledge_graph.tests.test_models import _artifact

pytestmark = pytest.mark.django_db(transaction=True)


def test_graph_model_state_matches_reserved_migration():
    from django.core.management import call_command

    call_command(
        "makemigrations", "apps_knowledge_graph", check=True, dry_run=True, verbosity=0
    )


@pytest.mark.parametrize("bound", [False, True])
def test_real_migration_reverse_preserves_legacy_and_refuses_bound(monkeypatch, bound):
    fixture = ScopeFixture(monkeypatch)
    snapshot = RepresentationSnapshotV1((fixture.snapshot.selections[0],))
    ref = snapshot.selections[0].document
    artifact = _artifact(
        scope_id=ref.id,
        source_hash=ref.source_hash,
        **(binding_values(snapshot) if bound else {}),
    )
    artifact.save()
    current = MigrationExecutor(connection).loader.graph.leaf_nodes()
    try:
        if bound:
            with pytest.raises(IrreversibleError):
                MigrationExecutor(connection).migrate(
                    [("apps_knowledge_graph", "0012_contextual_candidate_methods")]
                )
            artifact.refresh_from_db()
            assert (
                artifact.representation_digest
                == binding_values(snapshot)["representation_digest"]
            )
        else:
            MigrationExecutor(connection).migrate(
                [("apps_knowledge_graph", "0012_contextual_candidate_methods")]
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM apps_knowledge_graph_graphartifact "
                    "WHERE id = %s",
                    [artifact.pk],
                )
                assert cursor.fetchone()[0] == 1
    finally:
        MigrationExecutor(connection).migrate(current)
    artifact.refresh_from_db()
    assert (artifact.representation_digest is not None) == bound


@pytest.mark.parametrize(
    "payload,digest", [(None, "a" * 64), ({}, None), ({}, "A" * 64)]
)
def test_database_rejects_unpaired_or_noncanonical_binding(payload, digest):
    artifact = _artifact()
    artifact.save()
    with pytest.raises(IntegrityError), transaction.atomic():
        QuerySet.update(
            GraphArtifact.objects.filter(pk=artifact.pk),
            representation_snapshot=payload,
            representation_digest=digest,
        )
