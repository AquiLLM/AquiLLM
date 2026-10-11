"""Authorization and requested membership guards for frozen dense ranking."""

from uuid import UUID

from apps.collections.services.retrieval_authorization import (
    revalidate_retrieval_authorization_context,
)

from .representation_contracts import SourceScopeUnavailable
from .representation_store import ordered_selections
from .source_loading import SourcePreparationLimited, current_source_runtime


def requested_dense_scope(snapshot, dense_snapshot, keys, requested_document_ids):
    original_ids = frozenset(s.document.id for s in ordered_selections(snapshot))
    if requested_document_ids is None:
        return original_ids, keys
    if (
        type(requested_document_ids) is not frozenset
        or any(type(value) is not UUID for value in requested_document_ids)
        or not requested_document_ids <= original_ids
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
    selected_keys = {
        s.generation_key
        for s in dense_snapshot.selections
        if s.document_id in requested_document_ids
    }
    return requested_document_ids, tuple(key for key in keys if key in selected_keys)


def _dense_runtime(snapshot, budget, using):
    runtime = current_source_runtime()
    if runtime is None:
        from .representation_manual_scope import private_acquisition_runtime
        from .representation_media_acquisition import _check_runtime
        runtime = private_acquisition_runtime()
        if runtime is None:
            raise SourcePreparationLimited("source_runtime_unavailable")
        _check_runtime(runtime)
        frozen = getattr(runtime, "_frozen_retrieval_scope", None)
        if (frozen is None or frozen.runtime is not runtime
            or frozen.authorization is not runtime.authorization
            or runtime.authorization is not getattr(runtime, "_media_authorization", None)
            or frozen.representation is not snapshot):
            raise SourceScopeUnavailable("source_scope_unavailable")
    if (
        runtime is None
        or runtime.budget is not budget
        or runtime.authorization is None
        or runtime.authorization.database_alias != using
    ):
        raise SourcePreparationLimited("source_runtime_unavailable")
    return runtime


def _authorized_runtime(snapshot, budget, using, *, runtime):
    if runtime is not _dense_runtime(snapshot, budget, using):
        raise SourceScopeUnavailable("source_scope_unavailable")
    if not budget.can_publish():
        raise SourcePreparationLimited("source deadline")
    current = revalidate_retrieval_authorization_context(context=runtime.authorization)
    if not budget.can_publish():
        raise SourcePreparationLimited("source deadline")
    if any(
        s.document.id not in current.document_ids
        or s.document.collection_id not in current.collection_ids
        for s in ordered_selections(snapshot)
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return runtime
