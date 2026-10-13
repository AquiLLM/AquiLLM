"""Persisted principal activity is checked within existing permission reads."""
import pytest
from django.contrib.auth import get_user_model
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.collections.services.retrieval_authorization import revalidate_retrieval_authorization_context
from apps.documents.services.chunk_search_scope import FrozenRetrievalScope
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.tests.representation_owner_predicate_support import legacy_case

pytestmark = pytest.mark.django_db(transaction=True)


def test_deactivated_persisted_principal_loses_selected_retrieval_scope():
    case = legacy_case()
    context = build_selected_scope_authorization_context(
        principal=case.user, selected_collection_ids=(case.collection.pk,), selected_documents=case.docs)
    assert context is not None
    scope = FrozenRetrievalScope(case.snapshot, None, None, context, "default", None,
                                 requested_document_ids=context.selected_document_ids)
    assert scope.validate() == ()
    get_user_model().objects.filter(pk=case.user.pk).update(is_active=False)
    assert case.user.is_active is True  # The captured Python instance is stale.
    with CaptureQueriesContext(connection) as queries:
        current = revalidate_retrieval_authorization_context(context=context)
    assert current.document_ids == ()
    assert current.collection_ids == ()
    assert len(queries) <= 2
    with pytest.raises(SourceScopeUnavailable, match="source_scope_unavailable"):
        scope.validate()
    assert build_selected_scope_authorization_context(
        principal=case.user, selected_collection_ids=(case.collection.pk,), selected_documents=case.docs) is None
