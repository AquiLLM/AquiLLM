"""Selected-document scalar validation for hybrid graph retrieval."""

from apps.collections.services.retrieval_authorization import (
    OpaquePrincipalReference,
    RetrievalAuthorizationContext,
    RetrievalReauthorizationCapability,
)
from apps.documents.services.hybrid_graph_authorization import (
    documents_match_retrieval_authorization,
)
from apps.documents.tests.test_chunk_search_candidates import (
    _DOC_A,
    _DOC_B,
    _DocumentWithoutLazyCollectionAccess,
)


def test_hybrid_documents_must_exactly_match_frozen_selected_scope() -> None:
    context = RetrievalAuthorizationContext(
        OpaquePrincipalReference("b" * 64),
        "default",
        "policy-v1",
        "c" * 64,
        frozenset((7, 9)),
        frozenset((_DOC_A, _DOC_B)),
        "d" * 64,
        RetrievalReauthorizationCapability(object(), object()),
    )
    doc_b = _DocumentWithoutLazyCollectionAccess(_DOC_B, 9)
    doc_a = _DocumentWithoutLazyCollectionAccess(_DOC_A, 7)

    assert documents_match_retrieval_authorization((doc_b, doc_a), context)
    assert not documents_match_retrieval_authorization((doc_a,), context)
    assert not documents_match_retrieval_authorization(
        (doc_a, _DocumentWithoutLazyCollectionAccess(_DOC_B, 7)),
        context,
    )
    assert not documents_match_retrieval_authorization((doc_a, doc_b), object())
