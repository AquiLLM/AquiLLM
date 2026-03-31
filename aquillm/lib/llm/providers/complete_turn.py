"""Single-turn conversation completion orchestration for LLMInterface."""
from __future__ import annotations

import uuid
from os import getenv
from typing import Any, Awaitable, Callable, Literal, Optional

from ..types.conversation import Conversation
from ..types.messages import AssistantMessage, LLM_Message, ToolMessage, UserMessage
from ..types.response import LLMResponse
from ..types.tools import dump_tool_choice
from . import fallback_heuristics as fb
from . import image_context as imgctx
from . import rag_citations as citations
from .summary import generate_compact_tool_summary

try:
    from aquillm.settings import DEBUG
except ImportError:
    DEBUG = False

if DEBUG:
    from pprint import pp


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(getenv(name, str(default)))
    except Exception:
        value = default
    return max(minimum, value)


def _build_sources_block(allowed_citations: set[str]) -> str:
    refs = sorted(allowed_citations)
    if not refs:
        return ""
    source_lines = "\n".join(f"- {ref}" for ref in refs)
    return f"Sources:\n{source_lines}"


def _select_source_refs_for_response(text: str, allowed_citations: set[str]) -> set[str]:
    used = {c for c in citations.extract_citations(text or "") if c in allowed_citations}
    if used:
        return used
    return allowed_citations


def _append_citation_sources_if_missing(
    text: str,
    allowed_citations: set[str],
) -> str:
    base = (text or "").rstrip()
    if not allowed_citations:
        return base
    if "Sources:" in base:
        return base
    source_refs = _select_source_refs_for_response(base, allowed_citations)
    sources_block = _build_sources_block(source_refs)
    if not sources_block:
        return base
    return f"{base}\n\n{sources_block}"


async def complete_conversation_turn(
    llm: Any,
    conversation: Conversation,
    max_tokens: int,
    stream_func: Optional[Callable[[dict], Awaitable[Any]]] = None,
) -> tuple[Conversation, Literal["changed", "unchanged"]]:
    """Complete one assistant turn: tools, LLM call, retries, and fallbacks."""
    if len(conversation) < 1:
        return conversation, "unchanged"
    system_prompt = conversation.system
    messages_for_bot = [
        message
        for message in conversation
        if not (isinstance(message, ToolMessage) and message.for_whom == "user")
    ]
    last_message = conversation[-1]
    message_dicts = [message.render(include={"role", "content"}) for message in messages_for_bot]
    if isinstance(last_message, ToolMessage) and last_message.for_whom == "user":
        return conversation, "unchanged"
    if isinstance(last_message, AssistantMessage):
        if last_message.tools and last_message.tool_call_id:
            new_tool_msg = llm.call_tool(last_message)
            return conversation + [new_tool_msg], "changed"
        return conversation, "unchanged"

    assert isinstance(last_message, (UserMessage, ToolMessage)), "Type assertion failed"
    is_post_tool_result_turn = (
        isinstance(last_message, ToolMessage) and last_message.for_whom == "assistant"
    )
    citation_allowlist: set[str] = set()
    enforce_citations = False
    request_system_prompt = system_prompt
    if is_post_tool_result_turn and citations.citation_enforcement_enabled():
        citation_allowlist = citations.collect_allowed_chunk_citations(conversation)
        if citation_allowlist:
            enforce_citations = True
            request_system_prompt = (
                f"{system_prompt}\n\n"
                f"{citations.build_citation_system_suffix(citation_allowlist)}"
            )
    use_live_citation_stream = bool(enforce_citations and callable(stream_func))
    effective_stream_func = stream_func
    if use_live_citation_stream and callable(stream_func):
        async def _live_citation_stream(payload: dict) -> Any:
            out = dict(payload)
            content = str(out.get("content", ""))
            stop_reason = str(out.get("stop_reason", "")).strip().lower()
            is_cutoff_done = stop_reason in {"length", "max_tokens"}
            if out.get("done") and (not is_cutoff_done):
                out["content"] = _append_citation_sources_if_missing(content, citation_allowlist)
            await stream_func(out)

        effective_stream_func = _live_citation_stream
    tool_step_max_tokens = _env_int("LLM_TOOL_STEP_MAX_TOKENS", 512, minimum=128)
    post_tool_max_tokens = _env_int("LLM_POST_TOOL_MAX_TOKENS", 1536, minimum=256)
    continuation_max_tokens = _env_int("LLM_CONTINUATION_MAX_TOKENS", 768, minimum=128)
    citation_retry_prior_max_chars = _env_int("LLM_CITATION_RETRY_PRIOR_MAX_CHARS", 2400, minimum=512)
    request_max_tokens = max_tokens
    if isinstance(last_message, UserMessage) and last_message.tools:
        request_max_tokens = min(max_tokens, tool_step_max_tokens)
    elif is_post_tool_result_turn:
        request_max_tokens = min(max_tokens, post_tool_max_tokens)
    if last_message.tools:
        tools = {
            "tools": [tool.llm_definition for tool in last_message.tools],
            "tool_choice": dump_tool_choice(last_message.tool_choice),
        }
    else:
        tools = {}
    stream_message_uuid = str(uuid.uuid4())
    sdk_args = {
        **(
            llm.base_args
            | tools
            | {
                "system": request_system_prompt,
                "messages": message_dicts,
                "messages_pydantic": messages_for_bot,
                "max_tokens": request_max_tokens,
                "stream_callback": effective_stream_func,
                "stream_message_uuid": stream_message_uuid,
            }
        )
    }

    response = await llm.get_message(**sdk_args)
    should_force_tool_retry = (
        bool(last_message.tools)
        and bool(last_message.tool_choice)
        and last_message.tool_choice.type == "auto"
        and not response.tool_call
        and fb.looks_like_deferred_tool_intent(response.text)
    )
    if should_force_tool_retry:
        retry_args = sdk_args | {"tool_choice": {"type": "any"}}
        response = await llm.get_message(**retry_args)

    if is_post_tool_result_turn:
        response_text_for_retry = (response.text or "").strip()
        response_has_tool_call = bool(response.tool_call)
        needs_final_synthesis_retry = (not response_has_tool_call and not response_text_for_retry) or (
            not response_has_tool_call and fb.looks_like_deferred_tool_intent(response.text)
        )
        if needs_final_synthesis_retry:
            finalize_prompt = (
                "Use the tool results above to answer the user's last request directly. "
                "Do not call tools. Return a complete final answer in plain text."
            )
            finalize_messages = message_dicts + [{"role": "user", "content": finalize_prompt}]
            finalize_pydantic_messages = messages_for_bot + [UserMessage(content=finalize_prompt)]
            finalize_args = llm.base_args | {
                "system": request_system_prompt,
                "messages": finalize_messages,
                "messages_pydantic": finalize_pydantic_messages,
                "max_tokens": min(max_tokens, post_tool_max_tokens),
                "stream_callback": effective_stream_func,
                "stream_message_uuid": stream_message_uuid,
            }
            response = await llm.get_message(**finalize_args)
        post_finalize_text = (response.text or "").strip()
        if (not response.tool_call) and (
            (not post_finalize_text)
            or fb.looks_like_deferred_tool_intent(post_finalize_text)
        ):
            compact_summary = await generate_compact_tool_summary(llm, conversation, max_tokens)
            if compact_summary:
                response = LLMResponse(
                    text=compact_summary,
                    tool_call={},
                    stop_reason="stop",
                    input_usage=response.input_usage,
                    output_usage=response.output_usage,
                    model=response.model,
                    message_uuid=response.message_uuid,
                )

    allowed_tool_names = {tool.name for tool in (last_message.tools or [])}
    response_text = response.text if response.text else ""
    response_tool_call = response.tool_call or {}

    if response_tool_call:
        called_tool_name = response_tool_call.get("tool_call_name")
        if (not allowed_tool_names) or (called_tool_name not in allowed_tool_names):
            response_tool_call = {}
            if not response_text.strip():
                compact_summary = await generate_compact_tool_summary(llm, conversation, max_tokens)
                response_text = compact_summary or (
                    fb.synthesize_from_recent_tool_results(conversation)
                    if fb.extractive_fallback_enabled()
                    else None
                ) or (
                    "I completed retrieval but received an unusable tool-call payload. "
                    "Please retry and I will provide a full summary."
                )

    if (not response_tool_call) and (not response_text.strip()):
        compact_summary = await generate_compact_tool_summary(llm, conversation, max_tokens)
        response_text = compact_summary or (
            fb.synthesize_from_recent_tool_results(conversation)
            if fb.extractive_fallback_enabled()
            else None
        ) or (
            "I retrieved supporting passages but could not generate a final answer. "
            "Please retry and I will provide a full summary."
        )

    stop_reason_normalized = str(response.stop_reason or "").strip().lower()
    if (
        (not response_tool_call)
        and response_text.strip()
        and stop_reason_normalized in {"length", "max_tokens"}
        and fb.looks_cut_off(response_text)
    ):
        preserve_partial_response = fb.should_preserve_cutoff_partial(response_text)
        continuation_response: Optional[LLMResponse] = None
        continuation_text = ""
        if fb.continue_on_cutoff_enabled():
            continuation_budget = min(max_tokens, post_tool_max_tokens, continuation_max_tokens)
            continuation_response = await llm._continue_cutoff_response(
                system_prompt=request_system_prompt,
                message_dicts=message_dicts,
                messages_for_bot=messages_for_bot,
                partial_text=response_text,
                max_tokens=continuation_budget,
                stream_callback=effective_stream_func,
                stream_message_uuid=stream_message_uuid,
            )
            if continuation_response is not None and not continuation_response.message_uuid:
                continuation_response.message_uuid = stream_message_uuid
            continuation_text = (
                (continuation_response.text or "").strip() if continuation_response else ""
            )
        if continuation_text and not fb.looks_like_deferred_tool_intent(continuation_text):
            separator = "\n" if response_text and not response_text.endswith(("\n", " ")) else ""
            response_text = f"{response_text.rstrip()}{separator}{continuation_text}"
            response = continuation_response
        elif not preserve_partial_response:
            compact_summary = await generate_compact_tool_summary(llm, conversation, max_tokens)
            if compact_summary:
                response_text = compact_summary
            elif fb.extractive_fallback_enabled():
                synthesized = fb.synthesize_from_recent_tool_results(conversation)
                if synthesized:
                    response_text = synthesized
    if enforce_citations and (not response_tool_call):
        is_streaming_turn = callable(stream_func)
        original_response_text = (response_text or "").strip()
        citations_valid = citations.response_has_required_citations(response_text, citation_allowlist)
        original_invalid = citations.find_invalid_citations(original_response_text, citation_allowlist)
        original_has_any_citation = bool(citations.extract_citations(original_response_text))
        should_soft_accept_original = (
            (not citations_valid)
            and (not original_invalid)
            and original_has_any_citation
            and fb.is_high_quality_summary(original_response_text)
        )
        if should_soft_accept_original:
            citations_valid = True
        if (not citations_valid) and (not is_streaming_turn):
            invalid = citations.find_invalid_citations(response_text, citation_allowlist)
            prior_for_retry = response_text
            if len(prior_for_retry) > citation_retry_prior_max_chars:
                prior_for_retry = prior_for_retry[:citation_retry_prior_max_chars].rstrip() + "\n[Truncated for citation retry.]"
            retry_prompt = citations.build_citation_retry_prompt(
                prior_answer=prior_for_retry,
                allowed_citations=citation_allowlist,
                invalid_citations=invalid,
            )
            retry_messages = message_dicts + [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": retry_prompt},
            ]
            retry_pydantic_messages = messages_for_bot + [
                AssistantMessage(content=response_text, stop_reason="stop"),
                UserMessage(content=retry_prompt),
            ]
            retry_args = llm.base_args | {
                "system": request_system_prompt,
                "messages": retry_messages,
                "messages_pydantic": retry_pydantic_messages,
                "max_tokens": min(max_tokens, post_tool_max_tokens),
                "stream_callback": effective_stream_func,
                "stream_message_uuid": stream_message_uuid,
            }
            retry_response = await llm.get_message(**retry_args)
            if retry_response and (not retry_response.tool_call):
                response = retry_response
                response_text = (retry_response.text or "").strip()
                response_tool_call = {}
        retry_invalid = citations.find_invalid_citations(response_text, citation_allowlist)
        if retry_invalid and (not citations.response_has_required_citations(response_text, citation_allowlist)):
            if is_streaming_turn:
                response_text = (
                    response_text.rstrip()
                    + "\n\n[Note: Some citation tokens could not be verified against retrieved chunks.]"
                )
            else:
                cited_extract = citations.synthesize_cited_extract_from_results(conversation)
                if cited_extract:
                    response_text = cited_extract
                    response_tool_call = {}
    if (
        (
            is_post_tool_result_turn
            or (
                isinstance(last_message, UserMessage)
                and imgctx.looks_like_image_display_request(last_message.content)
            )
        )
        and (not response_tool_call)
        and response_text.strip()
        and (not imgctx.contains_markdown_image(response_text))
    ):
        markdown_images = imgctx.recent_tool_image_markdown(conversation, max_images=3)
        if markdown_images:
            response_text = response_text.rstrip() + "\n\n" + "\n".join(markdown_images)
    if use_live_citation_stream and enforce_citations and (not response_tool_call):
        response_text = _append_citation_sources_if_missing(response_text, citation_allowlist)
    new_msg = AssistantMessage(
        content=response_text,
        stop_reason=response.stop_reason,
        tools=last_message.tools,
        tool_choice=last_message.tool_choice,
        usage=response.input_usage + response.output_usage,
        model=response.model,
        message_uuid=response.message_uuid or uuid.uuid4(),
        **response_tool_call,
    )
    if DEBUG:
        print("Response from LLM:")
        pp(new_msg.model_dump())

    return conversation + [new_msg], "changed"


__all__ = ["complete_conversation_turn"]
