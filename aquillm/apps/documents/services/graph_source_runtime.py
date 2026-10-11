"""Reuse original private graph accounting without enabling source text modes."""
from lib.llm.turn_context import current_turn
from .chunk_search_scope import current_scope
from .representation_contracts import SourceScopeUnavailable
from .representation_manual_scope import private_acquisition_runtime
from .representation_principal import runtime_principal
from .source_loading import SourceRuntime, current_source_runtime


def graph_source_runtime():
    public, turn, scope = current_source_runtime(), current_turn(), current_scope()
    runtime = public if public is not None else private_acquisition_runtime()
    if runtime is None:
        if scope is not None or getattr(turn, "_media_principal_binding", None) is not None:
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
        return None
    if type(runtime) is not SourceRuntime or runtime.authorization is None:
        raise SourceScopeUnavailable("graph_original_scope_unavailable")
    principal = runtime_principal(runtime, required=public is None)
    if principal is not None:
        from .representation_media_acquisition import _check_runtime
        from .representation_manual_scope import registered_child_record
        _check_runtime(runtime)
        registered_child_record(runtime)
    if turn is not None and runtime.budget is not turn.budget:
        raise SourceScopeUnavailable("graph_original_scope_unavailable")
    runtime.budget.check_active()
    frozen = getattr(runtime, "_frozen_retrieval_scope", None)
    if principal is not None or public is None:
        if (frozen is None or frozen.runtime is not runtime
            or frozen.authorization is not runtime.authorization
            or runtime.authorization is not getattr(runtime, "_media_authorization", None)):
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
    if public is None and scope is None:
        raise SourceScopeUnavailable("graph_original_scope_unavailable")
    if scope is not None:
        if (frozen is None or scope.runtime is not runtime
            or scope.authorization is not runtime.authorization
            or scope.using != runtime.authorization.database_alias
            or any(getattr(scope, name) is not getattr(frozen, name)
                   for name in ("representation", "dense", "space"))
            or not scope.requested_document_ids <= frozen.requested_document_ids):
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
    return runtime
