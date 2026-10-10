"""Actual frozen candidate-cache boundary binds and revalidates proposal policy."""
from dataclasses import replace

import pytest
from django.test import override_settings

from apps.chat.services.retrieval_query_proposals import build_query_proposals
from apps.chat.services.retrieval_query_runtime import _frozen_vocabulary, proposal_scope
from apps.collections.models import CollectionPermission
from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.collections.tests.test_vocabulary_scope import publish, service, entry
from apps.documents.services.chunk_search_scope import freeze_search_scope, bind_search_scope, reset_search_scope
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.documents.tests.test_embedding_generation import legacy as legacy, stage
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.django_db(transaction=True)
def test_candidate_cache_changes_with_query_policy_and_revokes_changed_pin(legacy, transport):
    from apps.documents.services.embedding_index_store import activate_embedding_generation
    doc = legacy[0]
    CollectionPermission.objects.create(user=doc.ingested_by, collection=doc.collection, permission="MANAGE")
    authorization = build_selected_scope_authorization_context(principal=doc.ingested_by,
        selected_collection_ids=(doc.collection_id,), selected_documents=[doc])
    _, lease, _, _ = stage(legacy, transport)
    activate_embedding_generation(lease, using="default")
    _, version = publish(doc.ingested_by, doc.collection, [entry("source", "instrument")])
    pin = service().attach_vocabulary(actor=doc.ingested_by, collection=doc.collection,
        version=version, expected_revision=0)
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authorization)
    with source_runtime_scope(runtime), override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key):
        snapshot = _frozen_vocabulary(runtime)
        row = build_query_proposals("source", vocabulary=snapshot)[1]
        scope = freeze_search_scope([doc], using="default", authorization=authorization)
        keys = []
        for value in (row, replace(row, policy_key="changed"), replace(row, text=row.text + " extra")):
            with proposal_scope(value, runtime=runtime):
                token = bind_search_scope(scope, value.text, graph_enabled=False)
                try:
                    keys.append(scope.cache_key())
                finally:
                    reset_search_scope(token)
        assert len(set(keys)) == 3
        with proposal_scope(row, runtime=runtime):
            token = bind_search_scope(scope, row.text, graph_enabled=False)
            try:
                pin.delete()
                with pytest.raises(ValueError, match="pins changed"):
                    scope.cache_key()
            finally:
                reset_search_scope(token)
