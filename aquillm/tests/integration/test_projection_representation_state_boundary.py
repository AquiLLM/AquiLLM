"""Actual restricted state-function publication and replay preserve original binding."""
import pytest
from django.db import connections
from django.utils import timezone

from apps.knowledge_graph.models import CollectionGraphProjection
from apps.knowledge_graph.projection.state_repository import StaleProjectionAuthority
from tests.integration.projection_boundary_support import (
    boundary_projection, claim, graph_configuration, graph_source, projection_roles,
    source_call, state_repository, upload,
)

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"])


def ready_parameters(row, *, owner="boundary-worker", valid=True):
    return [row.pk, row.collection_id, row.artifact_id, row.generation_key, owner,
            row.schema_version, row.projection_version, row.identifier_key_version,
            row.membership_epoch, row.membership_checksum, row.private_mapping_checksum,
            "a" * 64, "a" * 64, 0, 0, 0, 0, 0, 0, valid, timezone.now()]


def call_ready(row, **changes):
    values = ready_parameters(row, **changes)
    with connections["projection_state"].cursor() as cursor:
        cursor.execute("SELECT * FROM public.kg_projection_ready_compare_and_set("
                       + ",".join(["%s"] * len(values)) + ")", values)
        return cursor.fetchone()


@pytest.mark.parametrize("changes", [{"owner": None}, {"valid": None}, {"owner": "foreign"}])
def test_direct_state_role_cannot_publish_nullable_admission(boundary_projection, changes):
    _, row = boundary_projection
    claim(row)
    assert source_call(row) is True
    result = call_ready(row, **changes)
    assert result[0] is False
    row.refresh_from_db()
    assert row.state != "ready"


@pytest.mark.parametrize("changed", [False, True])
def test_actual_state_cas_checks_current_representation(boundary_projection, changed):
    from apps.documents.models import RawTextDocument
    source, row = boundary_projection
    claim(row)
    # This is the adversarial state-API composition control, not a Memgraph readiness fixture.
    assert source_call(row) is True
    if changed:
        RawTextDocument.objects.filter(pk=source.document.pk).update(full_text="new bytes")
    result = call_ready(row)
    assert result[0] is (not changed)
    row.refresh_from_db()
    assert row.state == ("superseded" if changed else "ready")


@pytest.mark.parametrize("changed", [False, True])
def test_locked_replay_retains_exact_persisted_artifact_binding(boundary_projection, changed):
    from apps.documents.models import RawTextDocument
    source, row = boundary_projection
    repository = state_repository()
    if changed:
        RawTextDocument.objects.filter(pk=source.document.pk).update(full_text="stale bytes")
    kwargs = dict(projection_id=row.pk, collection_id=row.collection_id,
                  artifact_id=row.artifact_id,
                  versions=(row.schema_version, row.projection_version, row.identifier_key_version),
                  now=timezone.now())
    if changed:
        with pytest.raises(StaleProjectionAuthority):
            repository.replay(**kwargs)
        row.refresh_from_db()
        assert row.state == "pending"
    else:
        key = repository.replay(**kwargs)
        replacement = CollectionGraphProjection.objects.get(pk=key)
        assert replacement.state == "pending"
        assert replacement.representation_snapshot == row.representation_snapshot
        assert replacement.representation_digest == row.representation_digest
        assert source_call(replacement) is False
        claim(replacement)
        assert source_call(replacement) is True

