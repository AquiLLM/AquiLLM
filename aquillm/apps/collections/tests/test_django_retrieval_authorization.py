"""Production selected-document retrieval authorization contracts."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from django.contrib.auth.models import User

from apps.collections.services.django_retrieval_authorization import (
    DjangoCollectionRetrievalPermissionPolicy,
    build_production_retrieval_authorization_context,
)
from apps.collections.services.retrieval_authorization import (
    RetrievalAuthorizationContext,
    revalidate_retrieval_authorization_context,
)

_DOC = UUID("11111111-1111-4111-8111-111111111111")


def _enable(settings) -> None:
    settings.KG_OVERLAY_ENABLED = True
    settings.KG_MEMGRAPH_TRAVERSAL_ENABLED = True
    settings.KG_GRAPH_DIRECT_ENABLED = True
    settings.KG_GRAPH_EXTENDED_ENABLED = True


def test_production_context_is_default_off_and_ignores_unselected_documents(
    monkeypatch, settings
):
    user = User(pk=7, username="selected-user")
    documents = (
        SimpleNamespace(id=_DOC, collection_id=3),
        SimpleNamespace(
            id=UUID("22222222-2222-4222-8222-222222222222"), collection_id=4
        ),
    )
    rows = [(3, _DOC)]
    monkeypatch.setattr(
        DjangoCollectionRetrievalPermissionPolicy,
        "current_authorized_document_scope",
        lambda self, **_kwargs: tuple(rows),
    )

    assert (
        build_production_retrieval_authorization_context(
            principal=user,
            selected_collection_ids=(3,),
            selected_documents=documents,
        )
        is None
    )

    _enable(settings)
    context = build_production_retrieval_authorization_context(
        principal=user,
        selected_collection_ids=(3,),
        selected_documents=documents,
    )

    assert type(context) is RetrievalAuthorizationContext
    assert context.selected_collection_ids == frozenset({3})
    assert context.selected_document_ids == frozenset({_DOC})


def test_production_context_reauthorization_observes_permission_revocation(
    monkeypatch, settings
):
    _enable(settings)
    user = User(pk=7, username="revoked-user")
    rows = [(3, _DOC)]
    monkeypatch.setattr(
        DjangoCollectionRetrievalPermissionPolicy,
        "current_authorized_document_scope",
        lambda self, **_kwargs: tuple(rows),
    )
    context = build_production_retrieval_authorization_context(
        principal=user,
        selected_collection_ids=(3,),
        selected_documents=(SimpleNamespace(id=_DOC, collection_id=3),),
    )
    assert type(context) is RetrievalAuthorizationContext

    rows.clear()
    current = revalidate_retrieval_authorization_context(context=context)

    assert current.collection_ids == ()
    assert current.document_ids == ()


def test_adaptive_only_uses_current_selected_scope_without_graph(monkeypatch, settings):
    from apps.chat.services.retrieval_authorization import (
        resolve_document_retrieval_authorization,
    )

    settings.KG_OVERLAY_ENABLED = False
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "adaptive")
    user = User(pk=7, username="adaptive-user")
    rows = [(3, _DOC)]
    monkeypatch.setattr(
        DjangoCollectionRetrievalPermissionPolicy,
        "current_authorized_document_scope",
        lambda self, **kw: tuple(rows),
    )
    docs = (
        SimpleNamespace(id=_DOC, collection_id=3),
        SimpleNamespace(id=UUID(int=2), collection_id=4),
    )
    context = resolve_document_retrieval_authorization(
        user, SimpleNamespace(collections=[3]), docs, None
    )
    assert context.selected_document_ids == frozenset({_DOC})
    rows.clear()
    assert not revalidate_retrieval_authorization_context(context=context).document_ids
    assert (
        resolve_document_retrieval_authorization(
            user, SimpleNamespace(collections=[3]), docs, object()
        )
        is None
    )
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "shadow")
    assert (
        resolve_document_retrieval_authorization(
            user, SimpleNamespace(collections=[3]), docs, None
        )
        is None
    )
