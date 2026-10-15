"""Metadata-only original selection capture, shared by retrieval and source tools."""
from dataclasses import replace
from uuid import UUID

from .representation_contracts import SourceScopeUnavailable


def original_document_metadata(runtime, *, using):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    from apps.collections.services.retrieval_authorization import revalidate_retrieval_authorization_context
    from .source_deadline import check_source_deadline

    authority = runtime.authorization
    ids = authority.selected_document_ids
    if using != authority.database_alias or any(type(value) is not UUID for value in ids):
        raise SourceScopeUnavailable("source_scope_unavailable")
    current = revalidate_retrieval_authorization_context(context=authority)
    if frozenset(current.document_ids) != ids:
        raise SourceScopeUnavailable("source_scope_unavailable")
    documents = []
    seen = set()
    for model in DESCENDED_FROM_DOCUMENT:
        check_source_deadline()
        remaining = len(ids) - len(documents)
        rows = model._base_manager.using(using).filter(id__in=ids).only(
            "pkid", "id", "collection_id", "full_text_hash",
        ).order_by("pkid")[:remaining + 1]
        for document in rows:
            if (document.id in seen or document.collection_id not in authority.selected_collection_ids
                    or document.collection_id not in current.collection_ids):
                raise SourceScopeUnavailable("source_scope_unavailable")
            seen.add(document.id)
            documents.append(document)
            if len(documents) > len(ids):
                raise SourceScopeUnavailable("source_scope_unavailable")
    if seen != ids:
        raise SourceScopeUnavailable("source_scope_unavailable")
    return tuple(documents)


def tool_scope(runtime, *, using, require_existing=False):
    from .chunk_search_scope import _capture_tool_scope
    from .source_loading import SourceRuntime, bounded_source_database, source_runtime_scope

    if not isinstance(runtime, SourceRuntime) or runtime.authorization is None:
        raise SourceScopeUnavailable("source_scope_unavailable")

    authority, budget, lock, cache = runtime.authorization, runtime.budget, runtime.lock, runtime.cache
    with lock:
        scope = getattr(runtime, "_frozen_retrieval_scope", None)
        if scope is None:
            if require_existing:
                raise SourceScopeUnavailable("source_scope_unavailable")
            def check():
                if not budget.can_publish():
                    raise SourceScopeUnavailable("source_scope_unavailable")
                if (runtime.authorization is not authority or runtime.budget is not budget
                        or runtime.lock is not lock or runtime.cache is not cache
                        or getattr(runtime, "_frozen_retrieval_scope", None) is not None):
                    raise SourceScopeUnavailable("source_scope_unavailable")
                if scope is not None and (scope.runtime is not runtime or scope.authorization is not authority
                        or scope.using != using or scope.requested_document_ids != authority.selected_document_ids
                        or frozenset(s.document.id for s in scope.representation.selections)
                        != authority.selected_document_ids):
                    raise SourceScopeUnavailable("source_scope_unavailable")

            check()  # Lock entry must not replace the original runtime identities.
            with source_runtime_scope(runtime):
                scope = _capture_tool_scope(runtime, using=using)
            check()  # Nested construction and its database cleanup have finished.
            with bounded_source_database(runtime, using):
                scope.validate()
            check()
            def install():
                check()
                runtime._frozen_retrieval_scope = replace(scope)
            if not budget.publish(install):
                raise SourceScopeUnavailable("source_scope_unavailable")
            return scope
        with bounded_source_database(runtime, using):
            scope = getattr(runtime, "_frozen_retrieval_scope", None)
            if scope is None:
                raise SourceScopeUnavailable("source_scope_unavailable")
            if scope.using != using or scope.runtime is not runtime:
                raise SourceScopeUnavailable("source_scope_unavailable")
            scope.validate()
            return replace(scope, requested_document_ids=frozenset(
                selected.document.id for selected in scope.representation.selections))


def current_tool_rows(queryset, *, runtime, require_existing=False):
    scope = tool_scope(runtime, using=queryset.db, require_existing=require_existing)
    return scoped_queryset(scope, queryset)


def scoped_queryset(scope, queryset):
    if queryset.query.combinator:
        raise SourceScopeUnavailable("source_scope_unavailable")
    low, high = queryset.query.low_mark, queryset.query.high_mark
    query = queryset.all()
    query.query.clear_limits()
    query = scope.queryset(query)
    query.query.set_limits(low, high)
    return query
