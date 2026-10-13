"""Body-deferred metadata stays inside the original authorized turn scope."""

from uuid import UUID

from django.apps import apps

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_media_acquisition import (
    _check_runtime,
    media_runtime,
)
from apps.documents.services.representation_tool_scope import tool_scope
from apps.documents.services.source_loading import (
    bounded_source_database,
    bounded_source_enabled,
    current_source_runtime,
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
    with (
        runtime.lock,
        bounded_source_database(runtime, runtime.authorization.database_alias),
    ):
        scope = tool_scope(runtime, using=runtime.authorization.database_alias)
        if (
            requested is not None
            and requested not in runtime.authorization.selected_document_ids
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")
        owners = []
        grouped = {}
        for selected in scope.representation.selections:
            ref = selected.document
            if requested is not None and requested != ref.id:
                continue
            model = apps.get_model(ref.model_label)
            owners.append((model, ref))
            grouped.setdefault(model, []).append(ref)
        loaded = {}
        for model, refs in grouped.items():
            deferred = [
                field.name
                for field in model._meta.concrete_fields
                if field.name in {"full_text", "extracted_caption"}
            ]
            rows = (
                model._base_manager.using(scope.using)
                .filter(pk__in=[ref.pkid for ref in refs])
                .defer(*deferred)
            )
            for row in rows:
                # PKs are model-local. UUID, collection and source revision must
                # match the same frozen owner, never independent membership sets.
                key = (model, row.pk, row.id, row.collection_id, row.full_text_hash)
                loaded[key] = row
        result = []
        for model, ref in owners:
            row = loaded.get(
                (model, ref.pkid, ref.id, ref.collection_id, ref.source_hash)
            )
            if row is None:
                raise SourceScopeUnavailable("source_scope_unavailable")
            result.append(row)
        scope.validate()
        return result
