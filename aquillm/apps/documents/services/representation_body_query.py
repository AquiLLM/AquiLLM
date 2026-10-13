"""One original metadata-selected body; full authority remains unchanged."""
from uuid import UUID

from .representation_contracts import SourceScopeUnavailable
from .representation_store import active_chunk_scope


def exact_document_queryset(scope, queryset, document_id):
    scope.validate()
    if type(document_id) is not UUID or document_id not in scope.requested_document_ids:
        raise SourceScopeUnavailable("source_scope_unavailable")
    return active_chunk_scope(
        queryset.filter(doc_id=document_id),
        snapshot=scope.representation,
        document_id=document_id,
    )
