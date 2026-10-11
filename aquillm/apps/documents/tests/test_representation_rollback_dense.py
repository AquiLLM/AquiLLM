"""Fresh representation rollback cannot borrow vectors from retired A or B."""
import pytest

from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.documents.services import embedding_index_query as query
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.embeddings.document_provider import observe_query_embedding
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .representation_support import retained as retained, upload as upload, revision
from .representation_rollback_support import active, build_dense, execute, frozen

pytestmark = pytest.mark.django_db(transaction=True)


def test_retained_parser_rollback_requires_new_compatible_dense_generation(retained, transport):
    doc, output, _ = retained
    original = output.extraction_revision
    space = transport[0]
    assert execute(doc, original.pk, original.parser_key)["documents"][0]["status"] == "activated"
    a = active(doc)
    snapshot_a, dense_a = build_dense(doc, space)
    ids_a = set(doc.chunks.values_list("pk", flat=True))
    b_revision = revision(output, "Corrected reading is 100 units.", "i6-dense-rollback-b")
    assert execute(doc, b_revision, "i6-dense-rollback-b")["documents"][0]["status"] == "activated"
    snapshot_b, dense_b = build_dense(doc, space)
    ids_b = set(doc.chunks.values_list("pk", flat=True))
    assert dense_a.generation_keys != dense_b.generation_keys
    assert snapshot_a.selections[0].generation_key != snapshot_b.selections[0].generation_key
    assert execute(doc, original.pk, original.parser_key)["documents"][0]["status"] == "activated"
    snapshot_a2 = frozen(doc)
    assert snapshot_a2.selections[0].generation_key != a.pk
    unavailable = query.freeze_dense_generations(snapshot_a2, space_key=space.space_key, using="default")
    assert unavailable.generation_keys == ()
    # The real freeze cannot certify identical text under the retired A's vectors.
    fresh_snapshot, fresh_dense = build_dense(doc, space)
    assert fresh_snapshot == snapshot_a2
    assert set(fresh_dense.generation_keys).isdisjoint(set(dense_a.generation_keys) | set(dense_b.generation_keys))
    auth = build_selected_scope_authorization_context(principal=doc.ingested_by,
        selected_collection_ids=(doc.collection_id,), selected_documents=(doc,))
    assert auth is not None
    runtime = SourceRuntime(TurnBudget(TurnLimits()), auth)
    observation = observe_query_embedding("reading", space=space, budget=runtime.budget)
    with source_runtime_scope(runtime):
        ranked = query.rank_compatible_chunks(snapshot=fresh_snapshot, dense_snapshot=fresh_dense,
            query=observation, limit=10, using="default", budget=runtime.budget)
    actual = {row.chunk_id for row in ranked}
    assert actual == set(doc.chunks.values_list("pk", flat=True)) and actual
    assert actual.isdisjoint(ids_a | ids_b)
