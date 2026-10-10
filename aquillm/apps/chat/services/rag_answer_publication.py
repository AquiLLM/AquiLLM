"""Application entry points for exact RAG publication on the existing turn."""

from asgiref.sync import sync_to_async

from lib.llm.answer_publication import (
    LIMITED_ANSWER,
    AnswerPublicationRejected,
    bind_answer,
)
from lib.llm.turn_context import current_turn

from .rag_answer_context import RagAnswerPublication as RagAnswerPublication
from .rag_publication_authority import _current_packet_authority
from .rag_publication_images import (
    capture_publication_images as capture_publication_images,
)
from .rag_publication_legacy import (
    _freeze_legacy_packet as _freeze_legacy_packet,
)
from .rag_publication_legacy import (
    bind_legacy_answer_packet as bind_legacy_answer_packet,
)
from .rag_publication_legacy import (
    legacy_tool_handoff as legacy_tool_handoff,
)


def bind_packet_publication(packet, *, lease=None, scope_check=None):
    turn = current_turn()
    if turn is None:
        return None
    context = None

    async def authorize():
        # Persistence calls back into its originating sync transaction via
        # AsyncToSync's current-thread executor. Channels' DB wrapper would call
        # close_old_connections inside that transaction and close its connection.
        return await sync_to_async(_current_packet_authority, thread_sensitive=True)(
            packet, context.remaining_seconds()
        )

    if turn.answer_publication is not None:
        turn.answer_publication.close()
    context = RagAnswerPublication(
        packet=packet,
        authorize=authorize,
        turn=turn,
        lease=lease,
        scope_check=scope_check or getattr(packet, "_publication_scope_check", None),
    )
    turn.answer_publication = context
    return context


async def finalize_rag_answer(text, *, packet, authorize, limited_reason=None):
    turn = current_turn()
    if turn is None:
        raise AnswerPublicationRejected("unbound_receipt")
    context = turn.answer_publication
    if context is None:
        context = RagAnswerPublication(
            packet=packet, authorize=authorize, turn=turn, lease=turn.budget._synthesis
        )
        turn.answer_publication = context
    if context.packet is not packet:
        raise AnswerPublicationRejected("evidence_changed")
    return await context.finalize(text, limited_reason=limited_reason)


async def limited_rag_result(convo, *, stream_func=None):
    from apps.chat.services.rag_evidence import EvidencePacket
    from lib.llm.providers.final_stream import send_final_answer_stream
    from lib.llm.types.messages import AssistantMessage, ToolMessage

    message = AssistantMessage(content=LIMITED_ANSWER, stop_reason="end_turn")
    state = current_turn()
    context = getattr(state, "answer_publication", None)
    if context is None and state is not None:
        packet = EvidencePacket([], [], [], "", "", "context_limited", "", 0)
        context = bind_packet_publication(packet, lease=state.budget._synthesis)
        context.historical_ids = frozenset(m.message_uuid for m in convo.messages)
    if context is not None:
        bind_answer(
            message,
            await context.finalize(LIMITED_ANSWER, limited_reason="unbound_evidence"),
            context,
        )
    else:
        message._rag_answer = True
    if convo.messages and isinstance(convo[-1], ToolMessage):
        convo = convo.model_copy(
            update={
                "messages": [
                    *convo.messages[:-1],
                    convo[-1].model_copy(
                        update={
                            "content": "Selected evidence unavailable for publication.",
                            "result_dict": {
                                "result": [],
                                "retrieval_status": "context_limited",
                            },
                        }
                    ),
                ]
            }
        )
    if stream_func:
        await send_final_answer_stream(
            stream_func,
            message_uuid=str(message.message_uuid),
            content=message.content,
            stop_reason=message.stop_reason,
            usage=0,
            message=message,
        )
    return convo + [message]
