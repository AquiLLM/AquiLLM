"""Shared initial completion shaping; no inference or publication authority."""

from ..types.conversation import Conversation
from ..types.messages import ToolMessage, UserMessage
from . import rag_citations as citations
from .complete_turn_policy import (
    DIRECT_SYNTHESIS_GROUNDING, _general_answer_max_tokens, _post_tool_global_max,
    _resolve_post_tool_max_tokens, _resolve_tool_step_max_tokens, turn_token_limits,
)
from .image_policy import CONDITIONAL_IMAGE_INSTRUCTION


def completion_prompt_inputs(conversation, *, rag_answer, post_tool, source_refs):
    system = conversation.system
    allowlist, enforce = set(), False
    if rag_answer and citations.citation_enforcement_enabled():
        allowlist = citations.collect_allowed_chunk_citations(
            Conversation(system="", messages=[conversation[-1]]))
        if allowlist:
            enforce = True
            system += f"\n\n{citations.build_citation_system_suffix(allowlist)}"
    if post_tool:
        if not rag_answer:
            instruction = (
                "Answer the user's request using the current tool result. "
                "Do not reuse document evidence or citations from earlier turns.")
        else:
            if DIRECT_SYNTHESIS_GROUNDING not in system:
                system += f"\n\n{DIRECT_SYNTHESIS_GROUNDING}"
            instruction = (
                "Final synthesis step: answer at the depth requested "
                "with detail proportionate "
                "to the selected evidence. Complete the supported answer"
                " without "
                "padding it. Do not emit status lines, tool markup, or "
                "promises to "
                "retrieve later.")
        system += f"\n\n{instruction}"
    if CONDITIONAL_IMAGE_INSTRUCTION not in system:
        system += f"\n\n{CONDITIONAL_IMAGE_INSTRUCTION}"
    sources = set(allowlist)
    if rag_answer and not sources:
        sources = source_refs
    return system, allowlist, sources, enforce


def initial_request_policy(conversation, max_tokens, *, stage, post_tool_cap):
    last = conversation[-1]
    post_tool = isinstance(last, ToolMessage) and last.for_whom == "assistant"
    choice = str(getattr(last.tool_choice, "type", "") or "").strip().lower()
    requested = max_tokens
    if isinstance(last, UserMessage) and last.tools:
        requested = _resolve_tool_step_max_tokens(max_tokens, choice)
    elif post_tool and stage != "direct_synthesis":
        requested = _resolve_post_tool_max_tokens(conversation,
            default_cap=post_tool_cap, global_max=_post_tool_global_max(max_tokens))
    if stage is None:
        stage = "tool_selection" if isinstance(last, UserMessage) and last.tools else (
            "post_tool_synthesis" if post_tool else "general_answer")
    if stage == "general_answer":
        cap = _general_answer_max_tokens()
        if cap > 0:
            requested = min(requested, cap)
    return requested, choice, stage


def protected_post_tool_inputs(conversation, max_tokens, *, stage):
    """Exact eligible initial text request; other paths conservatively abstain."""
    from lib.llm.turn_context import is_retrieval_tool
    from .complete_turn_sources import _collect_source_refs_from_tool_message

    if not conversation.messages:
        return None
    last = conversation[-1]
    if not isinstance(last, ToolMessage) or last.for_whom != "assistant" or last.tools:
        return None
    refs = _collect_source_refs_from_tool_message(last)
    if not is_retrieval_tool(last.tool_name) and not refs:
        return None
    system, _, _, _ = completion_prompt_inputs(
        conversation, rag_answer=True, post_tool=True, source_refs=refs)
    maximum, _, _ = initial_request_policy(conversation, max_tokens,
        stage=stage, post_tool_cap=turn_token_limits()[0])
    messages = [message.render(include={"role", "content"}) for message in conversation
                if not (isinstance(message, ToolMessage) and message.for_whom == "user")]
    return {"system_text": system, "message_list": messages, "max_tokens": maximum}
