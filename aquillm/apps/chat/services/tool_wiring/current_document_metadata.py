"""Body-deferred metadata stays inside the original authorized turn scope."""
from uuid import UUID

from django.apps import apps

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_media_acquisition import media_runtime, _check_runtime
from apps.documents.services.representation_tool_scope import tool_scope
from apps.documents.services.source_loading import (
    bounded_source_database, bounded_source_enabled, current_source_runtime,
)


def acquisition_metadata(document_id=None):
    public = current_source_runtime()
    runtime = public or media_runtime()
    if runtime is None:
        if bounded_source_enabled():
            raise SourceScopeUnavailable("source_scope_unavailable")
        return None
    if public is None:
        _check_runtime(runtime)
    requested = UUID(str(document_id)) if document_id is not None else None
    with runtime.lock, bounded_source_database(runtime, runtime.authorization.database_alias):
        scope = tool_scope(runtime, using=runtime.authorization.database_alias)
        if requested is not None and requested not in runtime.authorization.selected_document_ids:
            raise SourceScopeUnavailable("source_scope_unavailable")
        result = []
        for selected in scope.representation.selections:
            ref = selected.document
            if requested is not None and requested != ref.id:
                continue
            model = apps.get_model(ref.model_label)
            deferred = [field.name for field in model._meta.concrete_fields
                        if field.name in {"full_text", "extracted_caption"}]
            row = model._base_manager.using(scope.using).filter(
                pk=ref.pkid, id=ref.id, collection_id=ref.collection_id,
                full_text_hash=ref.source_hash).defer(*deferred).first()
            if row is None:
                raise SourceScopeUnavailable("source_scope_unavailable")
            result.append(row)
        scope.validate()
        return result
