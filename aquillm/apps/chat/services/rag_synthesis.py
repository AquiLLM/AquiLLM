"""Evidence-first synthesis for the direct RAG pipeline.

Given a conversation whose last turn is a (synthetic) tool result plus the matching
:class:`~apps.chat.services.rag_evidence.EvidencePacket`, produce a user-facing
assistant answer. This reuses the standard post-tool synthesis machinery
(:meth:`LLMInterface.complete`) so citation enforcement, image selection, and
retrieval-notice handling stay consistent with the tool loop, then layers
direct-RAG guarantees on top:

- ``no_results`` returns a transparent retrieval notice without an LLM call.
- Empty/unusable synthesis falls back to an extractive cited summary built from
  the packet (always on for direct RAG, independent of
  ``LLM_ALLOW_EXTRACTIVE_EVIDENCE_UI``).
- Figures are displayed only when the synthesized answer selects them.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from apps.chat.services.rag_config import synthesis_max_tokens
from apps.chat.services.rag_evidence import EvidencePacket
from apps.chat.services.rag_evidence_handoff import prepare_evidence_handoff
from apps.chat.services.rag_source_synthesis import (
    LIMITED_MESSAGE,
    complete_source_request,
    finalize_source_support,
    revalidate_source_packet,
)
from apps.documents.services.source_loading import (
    SourcePreparationLimited,
    current_source_runtime,
)
from lib.llm.evidence_guard import ContextLimited
from lib.llm.providers import visibility
from lib.llm.providers.rag_citations import _chunk_citation_from_row
from lib.llm.providers.retrieval_status import document_retrieval_notice
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage

logger = structlog.stdlib.get_logger(__name__)

_MAX_EXTRACTIVE_POINTS = 5


def _last_assistant_tool_result(convo: Conversation) -> ToolMessage | None:
    for msg in reversed(convo.messages):
        if isinstance(msg, ToolMessage) and msg.for_whom == "assistant":
            return msg
    return None


def _synthesis_unusable(content: str | None) -> bool:
    visible = visibility.strip_tool_markup(
        visibility.strip_thinking_blocks(content)
    ).strip()
    if not visible:
        return True
    if visibility.is_interim_assistant_text(visible):
        return True
    if visible == visibility.clean_response_failure_text(after_tool_result=True):
        return True
    return False


def _extractive_summary(packet: EvidencePacket) -> str:
    points: list[str] = []
    seen: set[tuple[str, str | None]] = set()
    for chunk in packet.chunks:
        text = chunk.get("text") or chunk.get("x") or ""
        boundary = re.search(r"[.!?](?:\s|$)", text)
        snippet = text[: boundary.start() + 1] if boundary else text[:240]
        if not snippet:
            continue
        citation = _chunk_citation_from_row(chunk)
        key = (snippet, citation)
        if key in seen:
            continue
        seen.add(key)
        if not citation:
            continue
        points.append(
            "\n".join(f"> {line} {citation}" for line in snippet.splitlines())
        )
        if not packet.source_mode and len(points) >= _MAX_EXTRACTIVE_POINTS:
            break
    if not points:
        return ""
    header = "Selected excerpts:"
    return header + "\n" + "\n".join(points)


def _no_results_notice(convo: Conversation, packet: EvidencePacket) -> str:
    notice = (packet.diagnostic_message or "").strip()
    last_tool = _last_assistant_tool_result(convo)
    if last_tool is not None:
        notice = notice or document_retrieval_notice(last_tool).strip()
    return notice or (
        f'I searched the {packet.search_scope} for "{packet.query}", '
        "but found no relevant passages."
    )


def _replace_last_content(
    convo: Conversation, last: AssistantMessage, new_content: str
) -> Conversation:
    updated = last.model_copy(update={"content": new_content})
    return Conversation(
        system=convo.system, messages=list(convo.messages[:-1]) + [updated]
    )


async def synthesize_from_evidence(
    llm_if: Any,
    convo: Conversation,
    packet: EvidencePacket,
    *,
    stream_func: Any = None,
    max_tokens: int | None = None,
) -> Conversation:
    """Produce and publish one exact final answer, including app fallbacks."""
    from apps.chat.services.rag_answer_publication import bind_packet_publication
    from lib.llm.answer_publication import LIMITED_ANSWER, bind_answer
    from lib.llm.providers.final_stream import send_final_answer_stream
    from lib.llm.turn_context import current_turn

    async def finish(result, *, limited=False):
        last = result[-1]
        if not isinstance(last, AssistantMessage) or last.tool_call_id:
            return result
        state = current_turn()
        context = getattr(state, "answer_publication", None)
        if context is None and state is not None:
            context = bind_packet_publication(packet, lease=state.budget._synthesis)
        if context is not None:
            answer = await context.finalize(
                last.content, limited_reason="unbound_evidence" if limited else None
            )
            bind_answer(last, answer, context)
        else:
            last.content = LIMITED_ANSWER
            last._rag_answer = True
        if last.content == LIMITED_ANSWER:
            # No newly selected source titles, excerpts or images accompany limits.
            for message in result.messages[len(convo.messages) - 1 : -1]:
                if isinstance(message, ToolMessage):
                    message.content = "Selected evidence unavailable for publication."
                    message.result_dict = {
                        "result": [],
                        "retrieval_status": "context_limited",
                    }
        if stream_func:
            await send_final_answer_stream(
                stream_func,
                message_uuid=str(last.message_uuid),
                content=last.content,
                stop_reason=last.stop_reason,
                usage=last.usage,
                message=last,
            )
        return result

    bounded_packet = packet.source_mode or (
        current_source_runtime() is not None and packet.source_authorization is not None
    )
    if bounded_packet:
        try:
            packet = await revalidate_source_packet(packet)
            packet = await finalize_source_support(packet)
        except SourcePreparationLimited:
            return await finish(
                convo
                + [AssistantMessage(content=LIMITED_MESSAGE, stop_reason="end_turn")],
                limited=True,
            )
    convo, request_convo = prepare_evidence_handoff(convo, packet)
    state = current_turn()
    if state is not None and state.answer_publication is not None:
        if state.answer_publication.packet is not packet:
            from lib.llm.answer_publication import AnswerPublicationRejected

            raise AnswerPublicationRejected("evidence_changed")
        state.answer_publication.historical_ids = frozenset(
            m.message_uuid for m in convo.messages
        )
    if packet.retrieval_status == "context_limited":
        return await finish(
            convo
            + [
                AssistantMessage(
                    content=LIMITED_MESSAGE
                    + (
                        "\n\n" + packet.diagnostic_message
                        if packet.diagnostic_message
                        else ""
                    ),
                    stop_reason="end_turn",
                )
            ],
            limited=True,
        )
    if packet.retrieval_status == "no_results" or not packet.chunks:
        return await finish(
            convo
            + [
                AssistantMessage(
                    content=_no_results_notice(convo, packet), stop_reason="end_turn"
                )
            ],
            limited=True,
        )

    budget = max_tokens if max_tokens is not None else synthesis_max_tokens()
    try:
        if bounded_packet:
            completed_request, _ = await complete_source_request(
                llm_if, request_convo, budget, None, packet=packet
            )
        else:
            from lib.llm.evidence_guard import EvidenceProtection, protect_evidence

            from .rag_context_budget import provider_context_capacity

            context, margin = provider_context_capacity(llm_if)
            state = current_turn()
            lease = None
            if state is not None and state.answer_publication is not None:
                from lib.llm.synthesis_dispatch import synthesis_limits
                from lib.retrieval.synthesis_budget import seal_synthesis

                lease = seal_synthesis(
                    state.budget,
                    request_convo[-1].content,
                    **synthesis_limits(budget),
                    authority=packet.source_authorization,
                    identities=tuple(
                        (c.chunk_id, c.source_fingerprint)
                        for c in packet.selection.candidates
                    )
                    if packet.selection
                    else (),
                )
                state.answer_publication.lease = lease
            protection = EvidenceProtection(
                request_convo[-1].content,
                context,
                budget,
                margin,
                require_context=False,
                allow_history_trim=True,
                turn_budget=state.budget if state else None,
                synthesis_lease=lease,
            )
            from lib.llm.turn_context import suppress_evidence_handoff

            with protect_evidence(protection), suppress_evidence_handoff():
                completed_request, _ = await llm_if.complete(
                    request_convo, budget, stream_func=None
                )
                if protection.limited_reason:
                    raise ContextLimited(protection.limited_reason)
    except Exception:
        return await finish(
            convo + [AssistantMessage(content=LIMITED_MESSAGE, stop_reason="end_turn")],
            limited=True,
        )
    # Only provider-created turns belong in stored history; request-only evidence
    # omission and provider context trimming must never rewrite earlier messages.
    result_convo = convo + list(completed_request.messages[len(convo) :])

    last = result_convo[-1]
    if not isinstance(last, AssistantMessage) or last.tool_call_id:
        return result_convo

    content = last.content or ""
    new_content = content
    if _synthesis_unusable(content):
        extractive = _extractive_summary(packet)
        if extractive:
            new_content = extractive

    if new_content != content:
        result_convo = _replace_last_content(result_convo, last, new_content)
    return await finish(result_convo)


__all__ = ["synthesize_from_evidence"]
