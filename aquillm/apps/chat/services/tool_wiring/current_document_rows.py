"""Private turn acquisition is bounded without changing legacy tool routing."""
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_media_acquisition import media_runtime
from apps.documents.services.representation_tool_scope import current_tool_rows
from apps.documents.services.source_loading import SourceRuntime, bounded_source_database, source_query_rows


def _runtime():
    runtime = media_runtime()
    if runtime is not None and (not isinstance(runtime, SourceRuntime) or runtime.authorization is None):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return runtime


def current_document_rows(query, *, complete=False):
    runtime = _runtime()
    if runtime is None:
        return tuple(query)
    with runtime.lock, bounded_source_database(runtime, query.db):
        query = current_tool_rows(query, runtime=runtime, require_existing=True)
        expected = query.count() if complete else None
        rows = source_query_rows(query, runtime=runtime, require_existing=True)
        if complete and (not rows or len(rows) != expected):
            raise SourceScopeUnavailable("complete_document_unavailable")
        return rows


def current_chunk_metadata(chunk_id):
    from apps.documents.models import TextChunk
    runtime = _runtime()
    if runtime is None:
        return None
    alias = runtime.authorization.database_alias
    with runtime.lock, bounded_source_database(runtime, alias):
        return current_tool_rows(TextChunk.objects.using(alias).all(), runtime=runtime,
            require_existing=True).filter(pk=chunk_id).only("id", "doc_id", "chunk_number").first()
