"""Metadata-only retained-media checks use the already sealed answer lifetime."""
from lib.llm.turn_context import current_turn
from .representation_contracts import SourceScopeUnavailable


def media_publication_context(runtime, capture):
    turn = current_turn()
    context = getattr(turn, "answer_publication", None)
    scope = getattr(runtime, "_frozen_retrieval_scope", None)
    if (turn is None or context is None or context.turn is not turn
        or turn.budget is not runtime.budget or context.budget is not runtime.budget
        or context.lease is None or context.lease is not runtime.budget._synthesis
        or context.lease.budget is not runtime.budget
        or context._media_runtime is not runtime
        or getattr(context.packet, "_publication_media_runtime", None) is not runtime
        or getattr(context.packet, "_media_bindings", None) is not context._media_bindings
        or not any(value is capture for value in context._media_bindings or ())
        or scope is None or scope.runtime is not runtime
        or scope.authorization is not runtime.authorization
        or scope.representation is not capture.snapshot
        or context._representation is not capture.snapshot
        or getattr(context.packet, "_publication_representation", None) is not capture.snapshot):
        raise SourceScopeUnavailable("media_publication_unavailable")
    context._active()
    context.lease.check_active()
    return context


def validate_published_capture(runtime, capture):
    from apps.chat.services.rag_publication_database import _bounded_database
    from apps.collections.services.retrieval_authorization import revalidate_retrieval_authorization_context
    from apps.documents.models import TextChunk
    from .representation_media_acquisition import _binding_metadata, validate_private_capture
    from .representation_store import active_chunk_scope, validate_representation_snapshot

    with runtime.lock:
        context = media_publication_context(runtime, capture)
        using = runtime.authorization.database_alias
        with _bounded_database(using, context.remaining_seconds()):
            validate_private_capture(runtime, capture)
            current = revalidate_retrieval_authorization_context(context=runtime.authorization)
            snapshot = validate_representation_snapshot(capture.snapshot, using=using)
            if any(item.document.id not in current.document_ids or
                   item.document.collection_id not in current.collection_ids for item in snapshot.selections):
                raise SourceScopeUnavailable("source_scope_unavailable")
            query = active_chunk_scope(TextChunk.objects.using(using).all(), snapshot=snapshot)
            binding, revision = _binding_metadata(query, capture.binding.chunk_id, using=using)
            if binding != capture.binding or revision != capture.source_fingerprint:
                raise SourceScopeUnavailable("media_capture_changed")
        media_publication_context(runtime, capture)
    return capture
