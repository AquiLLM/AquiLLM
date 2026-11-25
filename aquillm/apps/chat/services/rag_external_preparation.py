"""Accept native first-tool work at its existing outer caller boundary."""

from contextlib import contextmanager
from hashlib import sha256

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.turn_context import current_turn
from lib.llm.types.messages import ToolMessage, UserMessage

from .rag_action_tools import source_views
from .rag_recovery_origins import (
    _Origins, _accept_occurrence, _payload_rows, accept_recovery_action,
    origin_lock, recovery_action, start_recovery_origins,
)
from .rag_selection_coordinator import selection_question


def _question(consumer):
    conversation = consumer.convo
    user = next((m for m in reversed(conversation.messages) if isinstance(m, UserMessage)), None)
    return selection_question(conversation, user.content if user else "")


@contextmanager
def native_tool_preparation(consumer, message):
    """Yield acceptance for the actual outer result, never an inner future."""
    runtime, turn = current_source_runtime(), current_turn()
    if (message.tool_call_name not in {"vector_search", "search_single_document"}
            or runtime is None or turn is None):
        yield None
        return
    question = _question(consumer)
    with origin_lock(runtime):
        # An acquisition already owning this record keeps its own lifecycle.
        prior = getattr(runtime, "_recovery_origins", None)
        disposed = type(prior) is _Origins and prior.abandoned and not prior.occurrences
        origins = start_recovery_origins(question, turn.budget) if prior is None or disposed else None
    if origins is None:
        yield None
        return
    try:
        with recovery_action(origins) as action:
            def accept(result):
                # Auxiliary, error and durable replay results have no pending
                # native owner. Public labels cannot manufacture one.
                if not action.native and not action.preparation:
                    return
                if (type(result) is not ToolMessage or result.for_whom != "assistant"
                        or result.tool_name != message.tool_call_name
                        or "exception" in result.result_dict
                        or result.result_dict.get("retrieval_status") in {"uncertain", "context_limited"}):
                    return
                if current_turn() is not turn or _question(consumer) != question:
                    raise SourceScopeUnavailable("recovery_original_scope_unavailable")
                raw = result.result_dict
                with origin_lock(runtime):
                    accept_recovery_action(action, raw, source_views([raw]))
                    # This is the same existing record, not a transferable receipt.
                    runtime._external_preparation_origins = origins

            yield accept
    finally:
        if not origins.occurrences:
            # This context alone can accept its pending action. Invalidate its
            # empty record without waiting for an interrupted worker's lock.
            object.__setattr__(origins, "abandoned", True)
            lock = runtime.lock
            if lock.acquire(blocking=False):
                try:
                    if getattr(runtime, "_recovery_origins", None) is origins:
                        runtime._recovery_origins = None
                finally:
                    lock.release()


def resume_external_preparation(question, budget, initial_results, evidence_views):
    """Resume once, only for this turn's exact outer accepted result objects."""
    runtime = current_source_runtime()
    if runtime is None:
        return None
    origins = getattr(runtime, "_external_preparation_origins", None)
    if origins is None:
        return None
    with origin_lock(runtime):
        if getattr(runtime, "_external_preparation_origins", None) is not origins:
            raise SourceScopeUnavailable("recovery_original_scope_unavailable")
        runtime._external_preparation_origins = None
        if (type(origins) is not _Origins or origins is not getattr(runtime, "_recovery_origins", None)
                or origins.budget is not budget or origins.query != sha256(question.encode()).hexdigest()):
            raise SourceScopeUnavailable("recovery_original_scope_unavailable")
        origins.check()
        if not origins.occurrences or any(
                not any(result is occurrence.result for result in initial_results)
                or _payload_rows(occurrence.result) != occurrence.rows
                for occurrence in origins.occurrences):
            raise SourceScopeUnavailable("recovery_result_changed")
        for result in initial_results:
            if not any(occurrence.result is result for occurrence in origins.occurrences):
                _accept_occurrence(origins, result, evidence_views([result]), None)
        origins.check()
        return origins
