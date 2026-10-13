"""Real persisted graph semantics remain bound across frontier and hydration."""
import pytest
from django.db import connection

from apps.documents.models import TextChunk
from apps.knowledge_graph.models import CollectionRelationEvidence, RelationMention
from apps.knowledge_graph.retrieval import ready_materialization_v2 as materialization
from tests.integration.persisted_graph_acquisition_support import acquire_graph
from tests.integration.persisted_graph_gateway_support import owned_gateway
from tests.integration.persisted_graph_projection_support import (
    graph_configuration as graph_configuration, project_active_collection,
)
from tests.integration.persisted_graph_role_support import projection_roles as projection_roles
from tests.integration.persisted_graph_source_support import graph_source as graph_source, upload as upload
from tests.integration.test_persisted_graph_generation_boundary import FIRST, assert_actual_evidence

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"]
)


@pytest.mark.parametrize("changed", [False, True])
def test_same_persisted_row_requires_original_semantic_provenance(
    graph_source, projection_roles, monkeypatch, settings, tmp_path, changed,
):
    source = graph_source
    source.activate_and_build_document(FIRST)
    projection = project_active_collection(source)
    chunk = source.document.chunks.get()
    evidence = CollectionRelationEvidence.objects.get(artifact_id=projection.artifact_id)
    mention = RelationMention.objects.get(pk=evidence.relation_mention_id)
    original_confidence = mention.extraction_confidence
    new_confidence = 0.75 if original_confidence != 0.75 else 0.5
    rows, mutations = [], []
    real = materialization.provenance_for_chunk

    def provenance(**kwargs):
        value = real(**kwargs)
        rows.append(value)
        return value

    def barrier():
        if changed:
            # Deliberately bypass ORM immutability on this synthetic owned row:
            # a retained row ID alone must never authorize changed semantics.
            table = connection.ops.quote_name(RelationMention._meta.db_table)
            with connection.cursor() as cursor:
                cursor.execute(f"UPDATE {table} SET extraction_confidence=%s WHERE id=%s",
                               [new_confidence, mention.pk])
                assert cursor.rowcount == 1
            current = RelationMention.objects.get(pk=mention.pk)
            assert current.extraction_confidence == new_confidence
            assert current.chunk_id == chunk.pk and current.document_id == source.document.id
            assert CollectionRelationEvidence.objects.get(pk=evidence.pk).relation_mention_id == mention.pk
            mutations.append(True)

    monkeypatch.setattr(materialization, "provenance_for_chunk", provenance)
    with owned_gateway(tmp_path) as gateway:
        result, observed = acquire_graph(source, gateway, monkeypatch, settings, barrier=barrier)
    assert rows and observed.references and observed.after_swap_remaining_ms >= 1000
    assert not {"deadline", "frontier_budget"}.intersection(result.handoff.coverage.reasons)
    requested = {key for ref in observed.references for key in ref.evidence_provenance_keys}
    current = {key for row in rows for key in row}
    if changed:
        assert mutations == [True] and not requested <= current
        assert observed.inputs == () and result.handoff.candidates == ()
        assert result.selection.representation_snapshot is observed.representation
    else:
        assert not mutations and requested <= current
        assert_actual_evidence(result, observed, FIRST, chunk.representation_generation_id)
    assert TextChunk.objects.get(pk=chunk.pk).content == FIRST
