"""Request-local production authorization for document retrieval tools."""

from apps.collections.services.django_retrieval_authorization import (
    build_production_retrieval_authorization_context,
    build_selected_scope_authorization_context,
)
from apps.documents.services.hybrid_graph_authorization import (
    is_exact_authorization_context,
)


def resolve_document_retrieval_authorization(user, col_ref, documents, provided):
    if is_exact_authorization_context(provided):
        return provided
    if provided is not None:
        return None
    from apps.chat.services.rag_config import rag_preservation_config

    builder = (
        build_selected_scope_authorization_context
        if rag_preservation_config().active
        else build_production_retrieval_authorization_context
    )
    return builder(
        principal=user,
        selected_collection_ids=tuple(col_ref.collections),
        selected_documents=tuple(documents),
    )


__all__ = ["resolve_document_retrieval_authorization"]
