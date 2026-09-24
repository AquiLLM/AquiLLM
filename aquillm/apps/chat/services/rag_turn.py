"""One preservation lifetime per consumer user turn, never per tool/fallback."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from channels.db import database_sync_to_async
from django.db import connections

from apps.chat.services.rag_config import (
    direct_rag_max_queries,
    rag_preservation_config,
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
from lib.llm.turn_context import TurnContext, bind_turn
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
    if not rag_preservation_config().active:
        yield None
        return
    from apps.chat.services.rag_preservation_turn import normal_source_handoff

    parent = current_source_runtime()
    budget = (
        parent.budget
        if parent
        else TurnBudget(
            TurnLimits(actions=min(3, direct_rag_max_queries(), max_func_calls))
        )
    )
    task = asyncio.current_task()
    # A cancellation handle only: scope/runtime are request-local, not reusable state.
    consumer._active_evidence_turn = (task, budget)
    try:
        runtime = parent or await asyncio.wait_for(
            database_sync_to_async(create_runtime, thread_sensitive=False)(
                consumer, budget
            ),
            budget.remaining_ms() / 1000,
        )

        async def handoff(llm, convo, max_tokens, stream_func):
            try:
                return await normal_source_handoff(
                    consumer, llm, convo, max_tokens, stream_func
                )
            except Exception:
                from apps.chat.services.rag_source_synthesis import LIMITED_MESSAGE
                from lib.llm.types.messages import AssistantMessage

                budget.check_active()
                return convo + [
                    AssistantMessage(content=LIMITED_MESSAGE, stop_reason="end_turn")
                ], "changed"

        with (
            source_runtime_scope(runtime),
            bind_turn(
                TurnContext(
                    budget,
                    handoff,
                    max_func_calls=max_func_calls,
                    worker_cleanup=connections.close_all,
                )
            ),
        ):
            yield runtime
    except asyncio.CancelledError:
        budget.close("cancelled")
        raise
    finally:
        budget.close("finished")
        if getattr(consumer, "_active_evidence_turn", (None,))[0] is task:
            consumer._active_evidence_turn = None


def cancel_active_turn(consumer):
    active = getattr(consumer, "_active_evidence_turn", None)
    if active:
        task, budget = active
        budget.close("cancelled")
        task.cancel()
