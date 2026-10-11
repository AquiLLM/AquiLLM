"""Projection purpose filtering uses real current owners and restricted source reads."""
import pytest
from django.db import connections
from django.test.utils import CaptureQueriesContext

from apps.knowledge_graph.models import CollectionArtifactInput, CollectionGraphProjection
from apps.knowledge_graph.projection.django_projection_rows import DjangoProjectionOrmLoader
from apps.knowledge_graph.projection.worker import project_generation
from tests.integration.projection_boundary_support import (
    boundary_projection, graph_configuration, graph_source, projection_roles,
    state_repository, upload,
)

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"])


def test_default_orm_loader_routes_ready_audit_and_terminal_prune(boundary_projection):
    source, projection = boundary_projection
    assert project_generation(projection_id=projection.pk, lease_owner="boundary-worker").ready
    projection.refresh_from_db()
    loader = DjangoProjectionOrmLoader("projection_source", state_using="projection_source")
    with CaptureQueriesContext(connections["projection_state"]) as state_reads:
        ready = loader.load(projection_id=projection.pk, batch_size=10, purpose="audit")
    assert len(state_reads) == 0  # Function-only state role never receives table reads.
    assert ready["projection"]["state"] == "ready"
    assert ready["documents"] == ({
        "document_id": source.document.id,
        "artifact_id": CollectionArtifactInput.objects.get(artifact_id=projection.artifact_id).document_artifact_id,
    },)
    assert ready["chunks"]
    assert all(row["document_id"] == source.document.id for row in ready["chunks"])
    with pytest.raises(CollectionGraphProjection.DoesNotExist):
        loader.load(projection_id=projection.pk, batch_size=10, purpose="prune")

    from django.utils import timezone

    assert state_repository().supersede(projection_id=projection.pk, now=timezone.now())
    projection.refresh_from_db()
    assert projection.state == "superseded"
    with pytest.raises(CollectionGraphProjection.DoesNotExist):
        loader.load(projection_id=projection.pk, batch_size=10, purpose="audit")
    with CaptureQueriesContext(connections["projection_state"]) as state_reads:
        terminal = loader.load(projection_id=projection.pk, batch_size=10, purpose="prune")
    assert len(state_reads) == 0
    assert terminal["projection"]["state"] == "superseded"
    assert terminal["documents"] == ready["documents"]
    assert terminal["chunks"] == ready["chunks"]
