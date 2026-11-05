"""One preservation lifetime per consumer user turn, never per tool/fallback."""

import asyncio
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace

from channels.db import database_sync_to_async
from django.db import connections

from apps.chat.services.rag_config import (
    direct_rag_max_queries,
    rag_preservation_config,
    retrieval_timeout_ms,
)
from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.documents.services.source_loading import (
    SourceRuntime,
    bounded_source_database,
    current_source_runtime,
    source_runtime_scope,
)
from lib.llm.turn_context import TurnContext, bind_turn, current_turn
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def create_runtime(consumer, budget):
    from apps.documents.models.document import _get_document_types

    collections = tuple(consumer.col_ref.collections)
    runtime = SourceRuntime(budget, SimpleNamespace(database_alias="default"))
    with bounded_source_database(runtime, "default"):
        documents = [
            doc
            for model in _get_document_types()
            for doc in model.objects.filter(collection_id__in=collections).only(
                "id", "collection_id"
            )[:10000]
        ]
        runtime.authorization = build_selected_scope_authorization_context(
            principal=consumer.user,
            selected_collection_ids=collections,
            selected_documents=documents,
        )
    return runtime


@asynccontextmanager
async def preservation_turn(consumer, *, max_func_calls):
    from lib.evidence_observation import publish

    parent = current_source_runtime()
    parent_turn = current_turn()
    if parent_turn is not None:
        # Nested direct/manual/tool calls inherit the owner's lifetime and latch.
        yield parent
        return
    from .retrieval_query_runtime import query_proposals_enabled

    config = rag_preservation_config()
    active = config.active or query_proposals_enabled()
    budget = (
        parent.budget
        if parent
        else TurnBudget(
            TurnLimits(actions=min(3, direct_rag_max_queries(), max_func_calls),
                       retrieval_ms=retrieval_timeout_ms())
        )
    )
    task = asyncio.current_task()
    owns_budget = parent is None
    # A cancellation handle only: scope/runtime are request-local, not reusable state.
    consumer._active_evidence_turn = (task, budget)
    try:
        if not active:
            from apps.chat.services.rag_answer_publication import legacy_tool_handoff

            historical_ids = frozenset(
                m.message_uuid
                for m in getattr(getattr(consumer, "convo", None), "messages", ())
            )

            async def publication_handoff(llm, convo, max_tokens, stream_func):
                if current_turn().answer_publication is not None:
                    from lib.llm.answer_publication import AnswerPublicationRejected

                    raise AnswerPublicationRejected("evidence_changed")
                if convo[-1].message_uuid in historical_ids:
                    return None
                return await legacy_tool_handoff(
                    consumer, llm, convo, max_tokens, stream_func
                )

            # Acquisition fencing is independent of preservation/planner/synthesis.
            turn = TurnContext(
                    budget,
                    handoff=publication_handoff,
                    max_func_calls=max_func_calls,
                    worker_cleanup=connections.close_all,
                )
            from apps.documents.services.representation_media_acquisition import initialize_media_runtime
            await asyncio.wait_for(database_sync_to_async(initialize_media_runtime, thread_sensitive=False)(
                consumer, turn), budget.remaining_ms() / 1000)
            with bind_turn(turn):
                yield None
            return
        from apps.chat.services.rag_preservation_turn import normal_source_handoff

        async def handoff(llm, convo, max_tokens, stream_func):
            try:
                return await normal_source_handoff(
                    consumer, llm, convo, max_tokens, stream_func
                )
            except Exception:
                from apps.chat.services.rag_answer_publication import limited_rag_result

                budget.check_active()
                return (
                    await limited_rag_result(convo, stream_func=stream_func),
                    "changed",
                )

        turn = TurnContext(
                    budget,
                    handoff,
                    max_func_calls=max_func_calls,
                    worker_cleanup=connections.close_all,
                    initial_source_limit=(
                        budget.limits.unique_sources - budget.limits.unique_sources // 3
                        if (
                            config.iterative_retrieval_enabled and budget.limits.actions > 1
                            and budget.planner_calls < budget.limits.planner_calls
                        )
                        else None
                    ),
                )
        from apps.documents.services.representation_media_acquisition import initialize_media_runtime
        runtime = parent or await asyncio.wait_for(
            database_sync_to_async(initialize_media_runtime, thread_sensitive=False)(consumer, turn),
            budget.remaining_ms() / 1000,
        )
        authorized = isinstance(runtime, SourceRuntime) and runtime.authorization is not None
        with (
            source_runtime_scope(runtime) if authorized else nullcontext(),
            bind_turn(turn),
        ):
            yield runtime if authorized else None
    except asyncio.CancelledError:
        budget.close("cancelled")
        raise
    finally:
        if owns_budget:
            budget.close("finished")
        publish(
            "turn_complete",
            {
                "pairs": budget.pairs_used,
                "actions": budget.actions_used,
                "planner_calls": budget.planner_calls,
                "text": budget.text_used,
                "stop_reason": budget.stop_reason,
            },
        )
        if getattr(consumer, "_active_evidence_turn", (None,))[0] is task:
            consumer._active_evidence_turn = None


def cancel_active_turn(consumer):
    active = getattr(consumer, "_active_evidence_turn", None)
    if active:
        task, budget = active
        budget.close("cancelled")
        task.cancel()
