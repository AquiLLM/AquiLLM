"""Reuse the caller's budget and original authorization across search proposals."""

from contextlib import ExitStack
from threading import RLock

from django.db.models.base import ModelBase

from lib.llm.turn_context import current_turn

from .chunk_search_scope import (
    bind_search_scope,
    freeze_search_scope,
    reset_search_scope,
)
from .hybrid_graph_authorization import is_exact_authorization_context
from .representation_contracts import SourceScopeUnavailable
from .source_loading import SourceRuntime, current_source_runtime, source_runtime_scope

_HOLDER_LOCK = RLock()


def enter_search(model_cls, documents, authorization, query, graph_enabled):
    stack = ExitStack()
    if not isinstance(model_cls, ModelBase):
        return stack, None  # Non-ORM service adapters cannot acquire dense authority.
    runtime = current_source_runtime()
    turn = current_turn()
    try:
        if (
            turn
            and runtime is None
            and not is_exact_authorization_context(authorization)
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")
        if runtime is None and turn and is_exact_authorization_context(authorization):
            with _HOLDER_LOCK:
                runtime = getattr(turn, "_retrieval_source_runtime", None)
                if runtime is None:
                    runtime = SourceRuntime(turn.budget, authorization)
                    turn._retrieval_source_runtime = runtime
                if runtime.budget is not turn.budget:
                    raise SourceScopeUnavailable("source_scope_unavailable")
            stack.enter_context(source_runtime_scope(runtime))
        with ExitStack() as initialization:
            if runtime:
                if turn and runtime.budget is not turn.budget:
                    raise SourceScopeUnavailable("source_scope_unavailable")
                if not runtime.lock.acquire(
                    timeout=runtime.budget.remaining_ms() / 1000
                ):
                    raise SourceScopeUnavailable("source_scope_unavailable")
                initialization.callback(runtime.lock.release)
            scope = freeze_search_scope(
                tuple(documents),
                using=(
                    runtime.authorization.database_alias
                    if runtime
                    else model_cls.objects.db
                ),
                authorization=authorization,
            )
        from dataclasses import replace

        scope = replace(scope, bodies={}, dense_rows={}, lexical_ids=set())
        token = bind_search_scope(scope, query, graph_enabled=graph_enabled)
        stack.callback(reset_search_scope, token)
        return stack, scope
    except BaseException:
        stack.close()
        raise


def graph_preflight(overlay, hybrid, docs, authorization, retrieval_scope, freeze):
    """Preserve captured legacy graph behavior; V2 must own new representations."""
    from .chunk_search_candidates import CandidateScopeLimit
    from .hybrid_graph_authorization import documents_match_retrieval_authorization

    config, scope, status, documents = None, None, None, docs
    if retrieval_scope and any(
        s.generation_key is not None for s in retrieval_scope.representation.selections
    ):
        return config, scope, "miss", documents
    if overlay:
        try:
            from apps.knowledge_graph.retrieval import get_graph_expansion_config

            config = get_graph_expansion_config()
            if hybrid:
                if not documents_match_retrieval_authorization(docs, authorization):
                    status = "error"
            else:
                scope = freeze(docs, config)
                documents = scope.documents
        except CandidateScopeLimit:
            status, scope = "miss", None
        except Exception:
            status, config, scope = "error", None, None
    return config, scope, status, documents
