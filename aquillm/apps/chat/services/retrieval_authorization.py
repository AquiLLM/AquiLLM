"""Request-local production authorization for document retrieval tools."""

from apps.collections.services.django_retrieval_authorization import (
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
    from contextlib import nullcontext
    from types import SimpleNamespace

    from apps.documents.services.source_loading import bounded_source_database
    from lib.llm.turn_context import current_turn

    turn = current_turn()
    boundary = (
        bounded_source_database(SimpleNamespace(budget=turn.budget), "default")
        if turn
        else nullcontext()
    )
    with boundary:
        from apps.documents.services.representation_manual_scope import manual_acquisition_collections
        manual_collections = manual_acquisition_collections(documents)
        return build_selected_scope_authorization_context(
            principal=user,
            selected_collection_ids=tuple(col_ref.collections) if manual_collections is None else manual_collections,
            selected_documents=tuple(documents),
        )


__all__ = ["resolve_document_retrieval_authorization"]
