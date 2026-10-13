"""Requested collection identity never substitutes for represented authority."""
from .representation_contracts import SourceScopeUnavailable
from .representation_manual_scope import registered_child_record
from .representation_principal import runtime_principal
from .representation_unavailable_parent import normalized_selection


def requested_selection(runtime, consumer):
    """Validate the original witness without recapturing any source or scope."""
    from lib.llm.turn_context import current_turn
    from .source_loading import SourceRuntime
    if not isinstance(runtime, SourceRuntime) or runtime.authorization is None:
        raise SourceScopeUnavailable("source_scope_unavailable")
    original_parent = getattr(current_turn(), "_media_source_runtime", None) is runtime
    binding = runtime_principal(runtime, required=original_parent)
    child = registered_child_record(runtime)
    runtime.budget.check_active()
    selected = normalized_selection(consumer.col_ref.collections)
    if selected is None:
        raise SourceScopeUnavailable("source_scope_unavailable")
    selected = frozenset(selected)
    represented = runtime.authorization.selected_collection_ids
    if binding is None or child is not None:
        # Standalone and explicitly registered manual scopes keep the old rule.
        if selected != represented:
            raise SourceScopeUnavailable("source_scope_unavailable")
        return selected
    scope = getattr(runtime, "_frozen_retrieval_scope", None)
    if (binding.consumer is not consumer or consumer.col_ref is not binding.collection_ref
        or binding.requested_collection_ids is None
        or selected != frozenset(binding.requested_collection_ids)
        or getattr(binding.turn, "_media_source_runtime", None) is not runtime
        or getattr(runtime, "_media_authorization", None) is not runtime.authorization
        or scope is None or scope.runtime is not runtime
        or scope.authorization is not runtime.authorization
        or not represented.issubset(selected)):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return selected


def requested_selection_unchanged(runtime, consumer, selected):
    """Publication callbacks fail closed on an absent or changed witness."""
    try:
        return requested_selection(runtime, consumer) == selected
    except Exception:
        return False
