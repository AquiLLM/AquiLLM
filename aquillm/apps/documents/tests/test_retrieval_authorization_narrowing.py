"""Fresh single-document tool contexts can narrow only the original authority."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from django.test import override_settings

from apps.chat.services.retrieval_authorization import (
    resolve_document_retrieval_authorization,
)
from apps.collections.models import CollectionPermission
from apps.collections.services.retrieval_authorization import OpaquePrincipalReference
from apps.documents.models import TextChunk
from apps.documents.services import embedding_index_store as store
from apps.documents.services.chunk_search import text_chunk_search
from apps.documents.services.chunk_search_lifetime import enter_search
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.tests.test_compatible_candidate_narrowing import two_documents
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_embedding_generation import stage
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.turn_context import TurnContext, bind_turn, current_turn

pytestmark = pytest.mark.django_db(transaction=True)


def authorize(documents):
    doc = documents[0]
    return resolve_document_retrieval_authorization(
        doc.ingested_by,
        SimpleNamespace(collections=(doc.collection_id,)),
        documents,
        None,
    )


@override_settings(KG_OVERLAY_ENABLED=False)
@pytest.mark.parametrize("swap", [False, True])
def test_default_all_then_single_keeps_original_authority(legacy, transport, swap):
    other, other_ref, runtime = two_documents(legacy, transport)
    CollectionPermission.objects.create(
        user=other.ingested_by, collection=other.collection, permission="VIEW"
    )
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        bind_turn(TurnContext(runtime.budget)),
    ):
        original = authorize([legacy[0], other])
        lifetime, first = enter_search(
            TextChunk, [legacy[0], other], original, "query", False
        )
        lifetime.close()
        narrowed = authorize([other])
        assert (
            narrowed.authorization_context_signature
            != original.authorization_context_signature
        )
        if swap:
            _, replacement, _, _ = stage((other, other_ref), transport)
            store.activate_embedding_generation(replacement, using="default")
        result = text_chunk_search(
            TextChunk, "query", 1, [other], authorization_context=narrowed
        )
        assert [r.doc_id for r in result[0]] == ([] if swap else [other.id])
        holder = current_turn()._retrieval_source_runtime
        assert holder.authorization is original
        assert holder.budget is runtime.budget
        assert holder._frozen_retrieval_scope.dense is first.dense


@pytest.mark.parametrize(
    "field,value",
    [
        ("principal_reference", OpaquePrincipalReference("f" * 64)),
        ("database_alias", "other"),
        ("policy_version", "changed"),
        ("policy_checksum", "e" * 64),
        ("authorization_context_signature", "d" * 64),
    ],
)
def test_changed_or_malformed_narrowed_context_never_replaces_original(
    legacy, transport, field, value
):
    other, _, runtime = two_documents(legacy, transport)
    CollectionPermission.objects.create(
        user=other.ingested_by, collection=other.collection, permission="VIEW"
    )
    with bind_turn(TurnContext(runtime.budget)):
        original = authorize([legacy[0], other])
        lifetime, _ = enter_search(
            TextChunk, [legacy[0], other], original, "query", False
        )
        lifetime.close()
        invalid = replace(authorize([other]), **{field: value})
        with pytest.raises(SourceScopeUnavailable):
            enter_search(TextChunk, [other], invalid, "query", False)
        assert current_turn()._retrieval_source_runtime.authorization is original
