"""An otherwise selectable replacement never repairs the captured handoff."""

from copy import copy
from time import monotonic
from uuid import uuid4

import pytest
from django.utils import timezone

from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.knowledge_graph.models import (
    CollectionGraphProjection,
    ProjectionChunkReference,
)
from apps.knowledge_graph.projection.django_projection_source import (
    DjangoProjectionRowSource,
)
from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain
from apps.knowledge_graph.projection.records import CollectionGraphProjectionBundleV1
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.retrieval.ready_materialization_v2 import (
    materialize_graph_candidates_v2,
)
from apps.knowledge_graph.retrieval.ready_scope_repository_v2 import (
    select_ready_scope_v2,
)
from apps.knowledge_graph.tests.graph_runtime_fixture_v2 import persisted_runtime
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db


def test_new_valid_ready_projection_cannot_reauthorize_old_selection(monkeypatch):
    f, runtime, settings, _ = persisted_runtime(monkeypatch)
    budget = TurnBudget(TurnLimits())
    with source_runtime_scope(SourceRuntime(budget, f.authorization)):
        result = runtime.run_v2(
            query="A",
            authorization=f.authorization,
            representation_snapshot=f.snapshot,
            budget=budget,
            settings=settings,
            deadline=monotonic() + 5,
        )
        old = f.projections["A"]
        CollectionGraphProjection.objects.filter(pk=old.pk).update(
            state="superseded", superseded_at=timezone.now()
        )
        replacement = copy(old)
        replacement.pk, replacement.generation_key = uuid4(), uuid4()
        replacement.ready_at = timezone.now()
        replacement.save(force_insert=True)
        chunk = f.chunks["A"]
        ProjectionChunkReference.objects.create(
            projection=replacement,
            projection_chunk_key=f.codec.encode(
                ProjectionIdentifierDomain.CHUNK,
                generation=replacement.generation_key,
                source=chunk.pk,
            ).value,
            chunk=chunk,
            integer_chunk_pk=chunk.pk,
            document_uuid=chunk.doc_id,
            chunk_number=chunk.chunk_number,
        )
        source = DjangoProjectionRowSource(
            "default", identifier_key=b"secret", identifier_key_version="key-v1"
        )
        bundle = CollectionGraphProjectionBundleV1(
            **source.load_projection_rows(
                projection_id=replacement.pk, batch_size=100, purpose="audit"
            )
        )
        digest = projection_checksum(bundle)
        CollectionGraphProjection.objects.filter(pk=replacement.pk).update(
            graph_checksum=digest, snapshot_checksum=digest
        )
        fresh = select_ready_scope_v2(
            authorization=f.authorization,
            settings=settings,
            representation_snapshot=f.snapshot,
        )
        assert fresh.coverage.ready_documents == 2
        assert {p.generation_id for g in fresh.groups for p in g.scope.projections} != {
            p.generation_id
            for g in result.selection.groups
            for p in g.scope.projections
        }

        def forbidden(**kwargs):
            raise AssertionError("final materializer attempted reselection")

        monkeypatch.setattr(
            "apps.knowledge_graph.retrieval.ready_scope_repository_v2.select_ready_scope_v2",
            forbidden,
        )
        assert (
            materialize_graph_candidates_v2(
                selected=result.selection,
                refs=result.handoff.candidates,
                authorization=f.authorization,
            )
            == ()
        )
