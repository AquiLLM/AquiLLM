"""Representation changes are exact and retain old chunk citations."""

from uuid import uuid4

import pytest
from django.db import transaction

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture

pytestmark = pytest.mark.django_db


def test_activation_requires_outer_transaction_and_exact_old_source(monkeypatch):
    from apps.knowledge_graph.projection.representation_lifecycle import (
        prepare_document_representation_activation,
    )

    fixture = ScopeFixture(monkeypatch)
    ref = fixture.snapshot.selections[0].document
    with pytest.raises(SourceScopeUnavailable):
        prepare_document_representation_activation(
            ref,
            (7,),
            expected_source_hash="0" * 64,
            old_representation_key=None,
            new_representation_key=uuid4(),
            using="default",
        )


def test_nonnull_key_cannot_impersonate_legacy_null(monkeypatch):
    from apps.knowledge_graph.projection.representation_lifecycle import (
        prepare_document_representation_activation,
    )

    fixture = ScopeFixture(monkeypatch)
    ref = fixture.snapshot.selections[0].document
    with transaction.atomic(), pytest.raises(SourceScopeUnavailable):
        prepare_document_representation_activation(
            ref,
            (7,),
            expected_source_hash=ref.source_hash,
            old_representation_key=uuid4(),
            new_representation_key=uuid4(),
            using="default",
        )


def test_activation_rollback_retains_authority_and_durable_intents(monkeypatch):
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphProjectionOutbox,
        GraphRebuildRequest,
    )
    from apps.knowledge_graph.projection.representation_lifecycle import (
        prepare_document_representation_activation,
    )

    fixture = ScopeFixture(monkeypatch)
    projection = fixture.persist_ready("A")
    ref = fixture.snapshot.selections[0].document
    before = GraphProjectionOutbox.objects.count(), GraphRebuildRequest.objects.count()
    with pytest.raises(RuntimeError, match="rollback"):
        with transaction.atomic():
            changed = prepare_document_representation_activation(
                ref,
                (7,),
                expected_source_hash=ref.source_hash,
                old_representation_key=None,
                new_representation_key=uuid4(),
                using="default",
            )
            assert projection.artifact_id in changed
            assert (
                GraphArtifact.objects.get(pk=projection.artifact_id).status
                == "superseded"
            )
            assert GraphProjectionOutbox.objects.filter(
                projection=projection, operation="prune", state="pending"
            ).exists()
            assert GraphRebuildRequest.objects.count() == before[1] + 1
            raise RuntimeError("rollback")
    projection.refresh_from_db()
    assert projection.state == "ready"
    assert GraphArtifact.objects.get(pk=projection.artifact_id).status == "active"
    assert (
        GraphProjectionOutbox.objects.count(),
        GraphRebuildRequest.objects.count(),
    ) == before


def test_binding_payload_is_immutable_and_exact(monkeypatch):
    from django.core.exceptions import ValidationError

    from apps.documents.services.representation_contracts import (
        RepresentationSnapshotV1,
    )
    from apps.documents.services.representation_store import (
        representation_snapshot_digest,
    )
    from apps.knowledge_graph.projection.representation_binding import snapshot_payload
    from apps.knowledge_graph.tests.test_models import _artifact

    fixture = ScopeFixture(monkeypatch)
    snapshot = RepresentationSnapshotV1((fixture.snapshot.selections[0],))
    source = snapshot.selections[0].document
    artifact = _artifact(
        scope_id=source.id,
        source_hash=source.source_hash,
        representation_snapshot=snapshot_payload(snapshot),
        representation_digest=representation_snapshot_digest(snapshot),
    )
    artifact.save()
    artifact.representation_snapshot["selections"][0]["document"]["pkid"] += 1
    with pytest.raises(ValidationError):
        artifact.save()


def test_schema_fields_default_to_legacy_null():
    from apps.knowledge_graph.models import CollectionGraphProjection, GraphArtifact

    for model in (GraphArtifact, CollectionGraphProjection):
        assert model._meta.get_field("representation_snapshot").default is None
        assert model._meta.get_field("representation_digest").default is None


def test_v2_extraction_excludes_retained_inactive_chunks(monkeypatch):
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.extraction.pipeline import _ordered_chunks

    fixture = ScopeFixture(monkeypatch)
    document = fixture.documents["A"]
    fields = dict(
        doc_id=document.id,
        content="A",
        start_position=0,
        end_position=1,
        embedding=[0.0] * 1024,
    )
    current = TextChunk.objects.create(chunk_number=0, **fields)
    TextChunk.objects.create(
        chunk_number=1,
        representation_generation=uuid4(),
        **{**fields, "start_position": 1, "end_position": 2},
    )
    monkeypatch.setenv("KG_GRAPH_REMEDIATION_MODE", "v2")
    assert tuple(c.pk for c in _ordered_chunks(document.id)) == (current.pk,)


def test_projection_publication_rejects_changed_captured_source(monkeypatch):
    from apps.documents.models import RawTextDocument
    from apps.documents.services.representation_contracts import (
        RepresentationSnapshotV1,
    )
    from apps.documents.services.representation_store import (
        representation_snapshot_digest,
    )
    from apps.knowledge_graph.projection.representation_binding import snapshot_payload
    from apps.knowledge_graph.projection.representation_inputs import (
        validate_projection_current,
    )

    fixture = ScopeFixture(monkeypatch)
    projection = fixture.persist_ready("A")
    snapshot = RepresentationSnapshotV1((fixture.snapshot.selections[0],))
    projection.representation_snapshot = snapshot_payload(snapshot)
    projection.representation_digest = representation_snapshot_digest(snapshot)
    RawTextDocument.objects.filter(pk=fixture.documents["A"].pk).update(
        full_text="changed"
    )
    with pytest.raises(SourceScopeUnavailable):
        validate_projection_current(projection, using="default")


def test_reverse_migration_refuses_bound_provenance(monkeypatch):
    import importlib

    from django.apps import apps
    from django.db import connection
    from django.db.migrations.exceptions import IrreversibleError

    from apps.documents.services.representation_contracts import (
        RepresentationSnapshotV1,
    )
    from apps.knowledge_graph.projection.representation_inputs import binding_values
    from apps.knowledge_graph.tests.test_models import _artifact

    fixture = ScopeFixture(monkeypatch)
    snapshot = RepresentationSnapshotV1((fixture.snapshot.selections[0],))
    ref = snapshot.selections[0].document
    artifact = _artifact(
        scope_id=ref.id, source_hash=ref.source_hash, **binding_values(snapshot)
    )
    artifact.save()
    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0013_graph_representation_binding_v1"
    )
    with connection.schema_editor() as editor, pytest.raises(IrreversibleError):
        migration.refuse_bound_reverse(apps, editor)
