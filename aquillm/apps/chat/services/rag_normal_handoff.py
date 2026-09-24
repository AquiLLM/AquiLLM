"""Current normal-tool evidence enters common final selection exactly once."""

from channels.db import database_sync_to_async

from apps.documents.services.source_loading import current_source_runtime
from lib.llm.turn_context import bounded_retrieval
from lib.llm.types.messages import AssistantMessage, UserMessage

from .rag_action_tools import RETRIEVAL_TOOLS, source_views
from .rag_config import rag_preservation_config
from .rag_selection_coordinator import selection_question
from .rag_source_continuity import current_continuity_result
from .rag_source_continuity_turn import continuity_candidates
from .rag_source_synthesis import LIMITED_MESSAGE


async def normal_source_handoff(consumer, llm, convo, max_tokens, stream_func):
    from .rag_preservation_turn import finish_preservation

    current = convo[-1]
    if current.tool_name not in RETRIEVAL_TOOLS:
        return None
    runtime = current_source_runtime()
    raw = dict(current.result_dict or {})
    auxiliary_handoff = None
    if current.tool_name in {"whole_document", "more_context"}:
        from apps.chat.services.tool_wiring.source_tool_revalidation import (
            revalidate_source_tool_result,
        )

        raw = await bounded_retrieval(
            database_sync_to_async(
                revalidate_source_tool_result, thread_sensitive=False
            )(raw, user=consumer.user),
            runtime.budget,
        )
        auxiliary_handoff = (raw, consumer.user)
        views = source_views([raw])
        converted = await bounded_retrieval(
            database_sync_to_async(current_continuity_result, thread_sensitive=False)(
                views
            ),
            runtime.budget,
        )
        if rag_preservation_config().evidence_text_mode != "source":
            from apps.chat.consumers.utils import truncate_tool_text

            for row in converted["result"]:
                row["text"] = truncate_tool_text(row["text"])
        raw = converted | {k: v for k, v in raw.items() if k.startswith("_")}
    user_message = next(
        (m for m in reversed(convo.messages) if isinstance(m, UserMessage)), None
    )
    question = selection_question(convo, user_message.content if user_message else "")
    prior, notice = await continuity_candidates(
        convo,
        question,
        user=consumer.user,
        selected_scope=consumer.col_ref.collections,
        enabled=rag_preservation_config().followup_evidence_enabled,
    )
    if notice and not prior:
        return convo + [
            AssistantMessage(content=notice, stop_reason="end_turn")
        ], "changed"
    results = (prior,) if notice else ((raw, prior) if prior else (raw,))
    try:
        result = await finish_preservation(
            consumer,
            llm,
            convo,
            question=question,
            initial_results=results,
            initial_acquired=True,
            stream_func=stream_func,
            auxiliary_handoff=auxiliary_handoff,
            notice=notice,
            iterative=False if notice else None,
        )
    except Exception:
        result = convo + [
            AssistantMessage(content=LIMITED_MESSAGE, stop_reason="end_turn")
        ]
    return result, "changed"
