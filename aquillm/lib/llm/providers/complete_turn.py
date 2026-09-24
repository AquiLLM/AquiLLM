"""Single-turn conversation completion orchestration for LLMInterface."""

from __future__ import annotations

import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from ..types.conversation import Conversation
from ..types.messages import AssistantMessage, LLM_Message, ToolMessage, UserMessage
from ..types.response import LLMResponse
from ..types.tools import ToolChoice, dump_tool_choice
from . import fallback_heuristics as fb
from . import final_stream, visibility
from . import image_context as imgctx
from . import rag_citations as citations
from .complete_turn_continuation import (
    _continuation_separator as _continuation_separator,
)
from .complete_turn_continuation import (
    _largest_common_prefix as _largest_common_prefix,
)
from .complete_turn_continuation import (
    _largest_suffix_prefix_overlap as _largest_suffix_prefix_overlap,
)
from .complete_turn_continuation import (
    _repair_continuation_seam as _repair_continuation_seam,
)
from .complete_turn_continuation import (
    _suffix_prefix_overlap_threshold as _suffix_prefix_overlap_threshold,
)
from .complete_turn_continuation import (
    _trim_duplicate_continuation_prefix as _trim_duplicate_continuation_prefix,
)
from .complete_turn_policy import (
    DIRECT_SYNTHESIS_GROUNDING as DIRECT_SYNTHESIS_GROUNDING,
)
from .complete_turn_policy import (
    _auto_tool_followup_direct_retry_enabled,
)
from .complete_turn_policy import (
    _compact_summary_fallback_enabled as _compact_summary_fallback_enabled,
)
from .complete_turn_policy import (
    _conversation_used_whole_document as _conversation_used_whole_document,
)
from .complete_turn_policy import (
    _direct_answer_retry_max_tokens as _direct_answer_retry_max_tokens,
)
from .complete_turn_policy import (
    _env_int as _env_int,
)
from .complete_turn_policy import (
    _env_optional_cap as _env_optional_cap,
)
from .complete_turn_policy import (
    _extractive_evidence_ui_enabled as _extractive_evidence_ui_enabled,
)
from .complete_turn_policy import (
    _general_answer_max_tokens as _general_answer_max_tokens,
)
from .complete_turn_policy import (
    _post_tool_evidence_retry_enabled as _post_tool_evidence_retry_enabled,
)
from .complete_turn_policy import (
    _post_tool_global_max as _post_tool_global_max,
)
from .complete_turn_policy import (
    _post_tool_output_ceiling as _post_tool_output_ceiling,
)
from .complete_turn_policy import (
    _post_tool_synthesis_retry_count as _post_tool_synthesis_retry_count,
)
from .complete_turn_policy import (
    _resolve_continuation_max_tokens as _resolve_continuation_max_tokens,
)
from .complete_turn_policy import (
    _resolve_post_tool_max_tokens as _resolve_post_tool_max_tokens,
)
from .complete_turn_policy import (
    _resolve_tool_step_max_tokens as _resolve_tool_step_max_tokens,
)
from .complete_turn_policy import (
    _tool_call_retry_max_tokens as _tool_call_retry_max_tokens,
)
from .complete_turn_sources import (
    _append_citation_sources_if_missing as _append_citation_sources_if_missing,
)
from .complete_turn_sources import (
    _build_sources_block as _build_sources_block,
)
from .complete_turn_sources import (
    _collect_doc_refs_from_embedded_images as _collect_doc_refs_from_embedded_images,
)
from .complete_turn_sources import (
    _collect_source_refs_from_tool_message as _collect_source_refs_from_tool_message,
)
from .complete_turn_sources import (
    _doc_ref as _doc_ref,
)
from .complete_turn_sources import (
    _extract_doc_ref_from_image_url as _extract_doc_ref_from_image_url,
)
from .complete_turn_sources import (
    _latest_user_requested_image as _latest_user_requested_image,
)
from .complete_turn_sources import (
    _select_source_refs_for_response as _select_source_refs_for_response,
)
from .request_observability import (
    current_correlation_id,
    current_stage,
    new_correlation_id,
)
from .retrieval_status import (
    append_retrieval_notice_if_missing,
    document_retrieval_notice,
)
from .summary import generate_compact_tool_summary

try:
    from aquillm.settings import DEBUG
except ImportError:
    DEBUG = False

if DEBUG:
    from pprint import pp


_DOC_IMAGE_URL_RE = re.compile(r"/aquillm/document_image/([^/]+)/")
_MECHANICAL_TOOL_MAX_TOKENS = 256


def _visible_text_is_empty_or_interim(text: str | None) -> bool:
    visible = visibility.strip_tool_markup(
        visibility.strip_thinking_blocks(text)
    ).strip()
    return (not visible) or visibility.is_interim_assistant_text(visible)


def _deterministic_required_tool_call(
    *,
    last_message: UserMessage,
    stream_message_uuid: str,
    input_usage: int,
    output_usage: int,
    model: str | None,
) -> LLMResponse | None:
    """Use a safe first retrieval call for a required turn with only reasoning."""
    available = {tool.name: tool for tool in (last_message.tools or [])}
    if "vector_search" in available:
        query = " ".join((last_message.content or "").split())
        if not query:
            return None
        return LLMResponse(
            text=None,
            tool_call={
                "tool_call_id": str(uuid.uuid4()),
                "tool_call_name": "vector_search",
                "tool_call_input": {
                    "search_string": query,
                    "top_k": 10,
                },
            },
            stop_reason="tool_use",
            input_usage=input_usage,
            output_usage=output_usage,
            model=model,
            message_uuid=stream_message_uuid,
        )
    if "document_ids" in available:
        return LLMResponse(
            text=None,
            tool_call={
                "tool_call_id": str(uuid.uuid4()),
                "tool_call_name": "document_ids",
                "tool_call_input": {},
            },
            stop_reason="tool_use",
            input_usage=input_usage,
            output_usage=output_usage,
            model=model,
            message_uuid=stream_message_uuid,
        )
    return None


def _latest_user_turn(conversation: Conversation) -> UserMessage | None:
    for msg in reversed(conversation.messages):
        if isinstance(msg, UserMessage):
            return msg
    return None


def _has_prior_assistant_context(conversation: Conversation) -> bool:
    for msg in conversation.messages[:-1]:
        if isinstance(msg, AssistantMessage) and visibility.is_displayable_answer_text(
            msg.content
        ):
            return True
    return False


def _post_tool_synthesis_unsatisfied(text: str | None) -> bool:
    visible = visibility.strip_tool_markup(
        visibility.strip_thinking_blocks(text)
    ).strip()
    if not visible:
        return True
    if visibility.is_interim_assistant_text(visible):
        return True
    if fb.looks_like_post_tool_non_answer(visible):
        return True
    if len(visible) >= 80:
        return False
    return not visibility.is_displayable_answer_text(visible)


def _build_synthesis_retry_prompt(conversation: Conversation, attempt: int) -> str:
    from .synthesis_retry_prompt import build_synthesis_retry_prompt

    user_turn = _latest_user_turn(conversation)
    return build_synthesis_retry_prompt(
        (user_turn.content or "").strip() if user_turn else "",
        attempt,
        current_stage(),
        _latest_user_requested_image(conversation),
        _post_tool_evidence_retry_enabled(),
    )


async def _run_post_tool_synthesis_attempt(
    llm: Any,
    *,
    system_prompt: str,
    message_dicts: list[dict],
    messages_for_bot: list[LLM_Message],
    conversation: Conversation,
    post_tool_max_tokens: int,
    global_max_tokens: int,
    stream_callback: Callable[[dict], Awaitable[Any]] | None,
    stream_message_uuid: str,
    attempt: int,
    allow_tools: bool,
) -> LLMResponse:
    prompt = _build_synthesis_retry_prompt(conversation, attempt)
    retry_messages = message_dicts + [{"role": "user", "content": prompt}]
    retry_pydantic_messages = messages_for_bot + [UserMessage(content=prompt)]
    retry_args: dict[str, Any] = llm.base_args | {
        "_synthesis_phase": "recovery",
        "system": system_prompt,
        "messages": retry_messages,
        "messages_pydantic": retry_pydantic_messages,
        "max_tokens": _resolve_post_tool_max_tokens(
            conversation,
            default_cap=post_tool_max_tokens,
            global_max=global_max_tokens,
        ),
        "stream_callback": stream_callback,
        "stream_message_uuid": stream_message_uuid,
    }
    if allow_tools:
        user_turn = _latest_user_turn(conversation)
        if user_turn and user_turn.tools:
            retry_args["tools"] = [tool.llm_definition for tool in user_turn.tools]
            retry_args["tool_choice"] = dump_tool_choice(
                user_turn.tool_choice or ToolChoice(type="auto")
            )
    return await llm.get_message(**retry_args)


async def _run_plain_followup_answer_retry(
    llm: Any,
    *,
    system_prompt: str,
    message_dicts: list[dict],
    messages_for_bot: list[LLM_Message],
    max_tokens: int,
    stream_callback: Callable[[dict], Awaitable[Any]] | None,
    stream_message_uuid: str,
) -> LLMResponse:
    prompt = (
        "Your previous attempt did not produce a user-visible answer. "
        "Answer the latest user message directly using the "
        "conversation history. "
        "Do not call tools, do not promise to search or "
        "retrieve, and do not mention internal tooling."
    )
    return await llm.get_message(
        **(
            llm.base_args
            | {
                "system": system_prompt,
                "messages": message_dicts + [{"role": "user", "content": prompt}],
                "messages_pydantic": messages_for_bot + [UserMessage(content=prompt)],
                "max_tokens": max(max_tokens, _direct_answer_retry_max_tokens()),
                "stream_callback": stream_callback,
                "stream_message_uuid": stream_message_uuid,
            }
        )
    )


async def _run_reasoning_cutoff_answer_retry(
    llm: Any,
    *,
    system_prompt: str,
    message_dicts: list[dict],
    messages_for_bot: list[LLM_Message],
    max_tokens: int,
    stream_callback: Callable[[dict], Awaitable[Any]] | None,
    stream_message_uuid: str,
) -> LLMResponse:
    prompt = (
        "Your reasoning budget ended without a user-visible answer. "
        "Answer the latest user message directly now. Do not "
        "include hidden analysis, "
        "do not call tools, and do not mention the prior failed "
        "attempt."
    )
    retry_max_tokens = _direct_answer_retry_max_tokens() or max_tokens
    return await llm.get_message(
        **(
            llm.base_args
            | {
                "system": system_prompt,
                "messages": message_dicts + [{"role": "user", "content": prompt}],
                "messages_pydantic": messages_for_bot + [UserMessage(content=prompt)],
                "max_tokens": retry_max_tokens,
                "thinking_budget": 0,
                "stream_callback": stream_callback,
                "stream_message_uuid": stream_message_uuid,
            }
        )
    )


async def _run_hidden_tool_call_retry(
    llm: Any,
    *,
    system_prompt: str,
    message_dicts: list[dict],
    messages_for_bot: list[LLM_Message],
    tools: dict[str, Any],
    current_max_tokens: int,
    stream_callback: Callable[[dict], Awaitable[Any]] | None,
    stream_message_uuid: str,
) -> LLMResponse:
    prompt = (
        "The prior attempt did not produce a visible tool call. "
        "Call exactly one available tool now using valid JSON "
        "arguments. "
        "Do not answer in prose before the tool result is available."
    )
    return await llm.get_message(
        **(
            llm.base_args
            | tools
            | {
                "system": system_prompt,
                "messages": message_dicts + [{"role": "user", "content": prompt}],
                "messages_pydantic": messages_for_bot + [UserMessage(content=prompt)],
                "max_tokens": min(
                    current_max_tokens,
                    _MECHANICAL_TOOL_MAX_TOKENS,
                ),
                "thinking_budget": 0,
                "stream_callback": stream_callback,
                "stream_message_uuid": stream_message_uuid,
            }
        )
    )


async def _retry_post_tool_synthesis(
    llm: Any,
    conversation: Conversation,
    *,
    system_prompt: str,
    message_dicts: list[dict],
    messages_for_bot: list[LLM_Message],
    post_tool_max_tokens: int,
    global_max_tokens: int,
    stream_callback: Callable[[dict], Awaitable[Any]] | None,
    stream_message_uuid: str,
    initial_response: LLMResponse,
) -> LLMResponse:
    """
    Re-run full-context synthesis until satisfied or retries exhausted.
    May return a tool_call so spin() can gather more evidence.
    """
    response = initial_response
    retry_count = _post_tool_synthesis_retry_count()
    chunk_evidence = citations.collect_allowed_chunk_citations(conversation)
    for attempt in range(retry_count):
        if not _post_tool_synthesis_unsatisfied(response.text):
            break
        allow_tools = (
            _post_tool_evidence_retry_enabled()
            and attempt == retry_count - 1
            and not chunk_evidence
        )
        response = await _run_post_tool_synthesis_attempt(
            llm,
            system_prompt=system_prompt,
            message_dicts=message_dicts,
            messages_for_bot=messages_for_bot,
            conversation=conversation,
            post_tool_max_tokens=post_tool_max_tokens,
            global_max_tokens=global_max_tokens,
            stream_callback=stream_callback,
            stream_message_uuid=stream_message_uuid,
            attempt=attempt,
            allow_tools=allow_tools,
        )
        if response.tool_call:
            return response
    return response


async def _last_resort_evidence_answer(
    llm: Any,
    conversation: Conversation,
    max_tokens: int,
    *,
    stream_callback: Callable[[dict], Awaitable[Any]] | None = None,
    stream_message_uuid: str | None = None,
) -> str | None:
    """Optional fallbacks; default is synthesis-only (no chunk dumps in the UI)."""
    if _extractive_evidence_ui_enabled():
        if fb.extractive_fallback_enabled():
            synthesized = fb.synthesize_from_recent_tool_results(conversation)
            if synthesized:
                return synthesized
        doc_extract = citations.synthesize_doc_level_extract_from_results(conversation)
        if doc_extract:
            return doc_extract
        cited_extract = citations.synthesize_cited_extract_from_results(conversation)
        if cited_extract:
            return cited_extract
    if _compact_summary_fallback_enabled():
        return await generate_compact_tool_summary(
            llm,
            conversation,
            max_tokens,
            stream_callback=stream_callback,
            stream_message_uuid=stream_message_uuid,
        )
    return None


async def complete_conversation_turn(
    llm: Any,
    conversation: Conversation,
    max_tokens: int,
    stream_func: Callable[[dict], Awaitable[Any]] | None = None,
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
    message_dicts = [
        message.render(include={"role", "content"}) for message in messages_for_bot
    ]
    if isinstance(last_message, ToolMessage) and last_message.for_whom == "user":
        return conversation, "unchanged"
    if isinstance(last_message, AssistantMessage):
        if last_message.tools and last_message.tool_call_id:
            from lib.llm.turn_context import call_tool_async

            new_tool_msg = await call_tool_async(llm, last_message)
            return conversation + [new_tool_msg], "changed"
        return conversation, "unchanged"

    assert isinstance(last_message, (UserMessage, ToolMessage)), "Type assertion failed"
    is_post_tool_result_turn = (
        isinstance(last_message, ToolMessage) and last_message.for_whom == "assistant"
    )
    citation_allowlist: set[str] = set()
    source_allowlist: set[str] = set()
    enforce_citations = False
    response_from_compact_summary = False
    request_system_prompt = system_prompt
    if is_post_tool_result_turn and citations.citation_enforcement_enabled():
        citation_allowlist = citations.collect_allowed_chunk_citations(conversation)
        if citation_allowlist:
            enforce_citations = True
            request_system_prompt = (
                f"{system_prompt}\n\n"
                f"{citations.build_citation_system_suffix(citation_allowlist)}"
            )
    if is_post_tool_result_turn:
        if current_stage() == "direct_synthesis":
            if DIRECT_SYNTHESIS_GROUNDING not in request_system_prompt:
                request_system_prompt += f"\n\n{DIRECT_SYNTHESIS_GROUNDING}"
            synthesis_instruction = (
                "Final synthesis step: answer at the depth requested "
                "with detail proportionate "
                "to the selected evidence. Complete the supported answer"
                " without "
                "padding it. Do not emit status lines, tool markup, or "
                "promises to "
                "retrieve later."
            )
        else:
            synthesis_instruction = (
                "Final synthesis step: answer the user using retrieved "
                "document content "
                "already in this thread. Write a thorough, well-"
                "structured user-facing "
                "answer with enough detail to stand alone; finish every "
                "section and do "
                "not stop mid-sentence. Do not emit status lines, tool "
                "markup, or promises "
                "to retrieve later."
            )
        request_system_prompt = f"{request_system_prompt}\n\n{synthesis_instruction}"
    source_allowlist = set(citation_allowlist)
    if is_post_tool_result_turn and not source_allowlist:
        source_allowlist = _collect_source_refs_from_tool_message(last_message)
    defer_stream_until_final = (
        callable(stream_func) and final_stream.final_answer_streaming_enabled()
    )
    use_live_citation_stream = bool(
        source_allowlist
        and callable(stream_func)
        and (not defer_stream_until_final)
        and citations.citation_sources_append_enabled()
    )
    effective_stream_func = stream_func
    if use_live_citation_stream and callable(stream_func):

        async def _live_citation_stream(payload: dict) -> Any:
            out = dict(payload)
            content = str(out.get("content", ""))
            stop_reason = str(out.get("stop_reason", "")).strip().lower()
            is_cutoff_done = stop_reason in {"length", "max_tokens"}
            if (
                out.get("done")
                and (not is_cutoff_done)
                and visibility.should_append_citation_sources(content)
            ):
                out["content"] = _append_citation_sources_if_missing(
                    content, source_allowlist
                )
            await stream_func(out)

        effective_stream_func = _live_citation_stream
    provider_stream_func = None if defer_stream_until_final else effective_stream_func
    post_tool_max_tokens = _env_int("LLM_POST_TOOL_MAX_TOKENS", 8192, minimum=256)
    continuation_max_tokens = _env_int("LLM_CONTINUATION_MAX_TOKENS", 4096, minimum=128)
    citation_retry_prior_max_chars = _env_int(
        "LLM_CITATION_RETRY_PRIOR_MAX_CHARS", 2400, minimum=512
    )
    observability_stage = current_stage()
    request_max_tokens = max_tokens
    tool_choice_type = (
        str(getattr(last_message.tool_choice, "type", "") or "").strip().lower()
    )
    if isinstance(last_message, UserMessage) and last_message.tools:
        request_max_tokens = _resolve_tool_step_max_tokens(max_tokens, tool_choice_type)
    elif is_post_tool_result_turn and observability_stage != "direct_synthesis":
        request_max_tokens = _resolve_post_tool_max_tokens(
            conversation,
            default_cap=post_tool_max_tokens,
            global_max=_post_tool_global_max(max_tokens),
        )
    if last_message.tools:
        tools = {
            "tools": [tool.llm_definition for tool in last_message.tools],
            "tool_choice": dump_tool_choice(last_message.tool_choice),
        }
    else:
        tools = {}
    stream_message_uuid = str(uuid.uuid4())
    observability_correlation_id = current_correlation_id() or new_correlation_id()
    if observability_stage is None:
        if isinstance(last_message, UserMessage) and last_message.tools:
            observability_stage = "tool_selection"
        elif is_post_tool_result_turn:
            observability_stage = "post_tool_synthesis"
        else:
            observability_stage = "general_answer"
    if observability_stage == "general_answer":
        general_answer_cap = _general_answer_max_tokens()
        if general_answer_cap > 0:
            request_max_tokens = min(request_max_tokens, general_answer_cap)
    sdk_args = {
        **(
            llm.base_args
            | tools
            | {
                "system": request_system_prompt,
                "messages": message_dicts,
                "messages_pydantic": messages_for_bot,
                "max_tokens": request_max_tokens,
                "stream_callback": provider_stream_func,
                "stream_message_uuid": stream_message_uuid,
            }
        )
    }
    supports_request_observability = bool(
        getattr(llm, "supports_request_observability", False)
    )
    if supports_request_observability:
        sdk_args["_observability_correlation_id"] = observability_correlation_id
        sdk_args["_observability_stage"] = observability_stage
    if isinstance(last_message, UserMessage) and last_message.tools:
        sdk_args["thinking_budget"] = 0

    response = await llm.get_message(**sdk_args)
    is_required_tool_cutoff = (
        isinstance(last_message, UserMessage)
        and bool(last_message.tools)
        and bool(last_message.tool_choice)
        and tool_choice_type == "any"
        and not response.tool_call
        and str(response.stop_reason or "").strip().lower() in {"length", "max_tokens"}
    )
    if is_required_tool_cutoff:
        deterministic_response = _deterministic_required_tool_call(
            last_message=last_message,
            stream_message_uuid=stream_message_uuid,
            input_usage=response.input_usage,
            output_usage=response.output_usage,
            model=response.model,
        )
        if deterministic_response is not None:
            response = deterministic_response
    should_force_tool_retry = (
        bool(last_message.tools)
        and bool(last_message.tool_choice)
        and tool_choice_type in {"auto", "any"}
        and not response.tool_call
        and fb.looks_like_deferred_tool_intent(response.text)
    )
    if should_force_tool_retry:
        retry_args = sdk_args | {
            "tool_choice": {"type": "any"},
            "max_tokens": min(request_max_tokens, _MECHANICAL_TOOL_MAX_TOKENS),
            "thinking_budget": 0,
        }
        if supports_request_observability:
            retry_args["_observability_stage"] = "tool_retry"
        response = await llm.get_message(**retry_args)

    if (
        isinstance(last_message, UserMessage)
        and bool(last_message.tools)
        and bool(last_message.tool_choice)
        and tool_choice_type == "any"
        and not response.tool_call
        and _visible_text_is_empty_or_interim(response.text)
    ):
        response = await _run_hidden_tool_call_retry(
            llm,
            system_prompt=request_system_prompt,
            message_dicts=message_dicts,
            messages_for_bot=messages_for_bot,
            tools=tools,
            current_max_tokens=request_max_tokens,
            stream_callback=provider_stream_func,
            stream_message_uuid=stream_message_uuid,
        )

    if (
        isinstance(last_message, UserMessage)
        and bool(last_message.tools)
        and bool(last_message.tool_choice)
        and tool_choice_type == "any"
        and not response.tool_call
        and _visible_text_is_empty_or_interim(response.text)
    ):
        deterministic_response = _deterministic_required_tool_call(
            last_message=last_message,
            stream_message_uuid=stream_message_uuid,
            input_usage=response.input_usage,
            output_usage=response.output_usage,
            model=response.model,
        )
        if deterministic_response is not None:
            response = deterministic_response

    _available_tool_names = {tool.name for tool in (last_message.tools or [])}
    if (
        isinstance(last_message, UserMessage)
        and bool(last_message.tools)
        and tool_choice_type == "auto"
        and "vector_search" in _available_tool_names
        and not response.tool_call
        and _visible_text_is_empty_or_interim(response.text)
    ):
        deterministic_response = _deterministic_required_tool_call(
            last_message=last_message,
            stream_message_uuid=stream_message_uuid,
            input_usage=response.input_usage,
            output_usage=response.output_usage,
            model=response.model,
        )
        if deterministic_response is not None:
            response = deterministic_response

    if is_post_tool_result_turn and not response.tool_call:
        response = await _retry_post_tool_synthesis(
            llm,
            conversation,
            system_prompt=request_system_prompt,
            message_dicts=message_dicts,
            messages_for_bot=messages_for_bot,
            post_tool_max_tokens=post_tool_max_tokens,
            global_max_tokens=_post_tool_global_max(max_tokens),
            stream_callback=provider_stream_func,
            stream_message_uuid=stream_message_uuid,
            initial_response=response,
        )

    allowed_tool_names = {tool.name for tool in (last_message.tools or [])}
    response_text = visibility.strip_tool_markup(
        visibility.strip_thinking_blocks(response.text)
    )
    response_tool_call = response.tool_call or {}
    initial_stop_reason = str(response.stop_reason or "").strip().lower()

    if response_tool_call:
        called_tool_name = response_tool_call.get("tool_call_name")
        if (not allowed_tool_names) or (called_tool_name not in allowed_tool_names):
            response_tool_call = {}
            if not response_text.strip():
                recovered = await _last_resort_evidence_answer(
                    llm,
                    conversation,
                    max_tokens,
                    stream_callback=provider_stream_func,
                    stream_message_uuid=stream_message_uuid,
                )
                response_text = recovered or (
                    "I completed retrieval but received an unusable tool-"
                    "call payload. "
                    "Please retry and I will provide a full summary."
                )

    should_retry_reasoning_only_cutoff = (
        observability_stage == "general_answer"
        and isinstance(last_message, UserMessage)
        and not last_message.tools
        and not response_tool_call
        and not response_text.strip()
        and initial_stop_reason in {"length", "max_tokens"}
    )
    if should_retry_reasoning_only_cutoff:
        retry_response = await _run_reasoning_cutoff_answer_retry(
            llm,
            system_prompt=request_system_prompt,
            message_dicts=message_dicts,
            messages_for_bot=messages_for_bot,
            max_tokens=request_max_tokens,
            stream_callback=provider_stream_func,
            stream_message_uuid=stream_message_uuid,
        )
        retry_text = visibility.strip_tool_markup(
            visibility.strip_thinking_blocks(retry_response.text)
        )
        if (
            not retry_response.tool_call
            and retry_text.strip()
            and not visibility.is_interim_assistant_text(retry_text)
        ):
            response = retry_response
            response_text = retry_text
            response_tool_call = {}

    should_plain_retry_auto_tool_followup = (
        _auto_tool_followup_direct_retry_enabled()
        and isinstance(last_message, UserMessage)
        and bool(last_message.tools)
        and tool_choice_type == "auto"
        and _has_prior_assistant_context(conversation)
        and (not response_tool_call)
        and (
            not response_text.strip()
            or visibility.is_interim_assistant_text(response_text)
        )
    )
    if should_plain_retry_auto_tool_followup:
        retry_response = await _run_plain_followup_answer_retry(
            llm,
            system_prompt=request_system_prompt,
            message_dicts=message_dicts,
            messages_for_bot=messages_for_bot,
            max_tokens=max_tokens,
            stream_callback=provider_stream_func,
            stream_message_uuid=stream_message_uuid,
        )
        retry_text = visibility.strip_tool_markup(
            visibility.strip_thinking_blocks(retry_response.text)
        )
        if (
            retry_response
            and (not retry_response.tool_call)
            and retry_text.strip()
            and (not visibility.is_interim_assistant_text(retry_text))
        ):
            response = retry_response
            response_text = retry_text
            response_tool_call = {}

    if (not response_tool_call) and visibility.is_interim_assistant_text(response_text):
        response_text = ""

    if (not response_tool_call) and (not response_text.strip()):
        recovered = await _last_resort_evidence_answer(
            llm,
            conversation,
            max_tokens,
            stream_callback=provider_stream_func,
            stream_message_uuid=stream_message_uuid,
        )
        if recovered and _compact_summary_fallback_enabled():
            response_from_compact_summary = True
        response_text = recovered or visibility.clean_response_failure_text(
            after_tool_result=is_post_tool_result_turn
        )

    stop_reason_normalized = str(response.stop_reason or "").strip().lower()
    if (
        (not response_tool_call)
        and response_text.strip()
        and stop_reason_normalized in {"length", "max_tokens"}
        and fb.looks_cut_off(response_text)
    ):
        preserve_partial_response = fb.should_preserve_cutoff_partial(response_text)
        continuation_response: LLMResponse | None = None
        continuation_text = ""
        if fb.continue_on_cutoff_enabled():
            post_tool_budget = _resolve_post_tool_max_tokens(
                conversation,
                default_cap=post_tool_max_tokens,
                global_max=_post_tool_global_max(max_tokens),
            )
            continuation_budget = _resolve_continuation_max_tokens(
                conversation,
                default_cap=continuation_max_tokens,
                post_tool_budget=post_tool_budget,
                global_max=_post_tool_global_max(max_tokens),
            )
            continuation_response = await llm._continue_cutoff_response(
                system_prompt=request_system_prompt,
                message_dicts=message_dicts,
                messages_for_bot=messages_for_bot,
                partial_text=response_text,
                max_tokens=continuation_budget,
                stream_callback=provider_stream_func,
                stream_message_uuid=stream_message_uuid,
            )
            if (
                continuation_response is not None
                and not continuation_response.message_uuid
            ):
                continuation_response.message_uuid = stream_message_uuid
            raw_continuation_text = (
                (continuation_response.text or "").strip()
                if continuation_response
                else ""
            )
            continuation_text = _trim_duplicate_continuation_prefix(
                response_text, raw_continuation_text
            )
            if raw_continuation_text and (not continuation_text):
                preserve_partial_response = True
        if continuation_text and not fb.looks_like_deferred_tool_intent(
            continuation_text
        ):
            separator = _continuation_separator(response_text, continuation_text)
            merged = f"{response_text.rstrip()}{separator}{continuation_text}"
            response_text = _repair_continuation_seam(response_text, merged)
            response = continuation_response
        elif not preserve_partial_response:
            retry_response = await _retry_post_tool_synthesis(
                llm,
                conversation,
                system_prompt=request_system_prompt,
                message_dicts=message_dicts,
                messages_for_bot=messages_for_bot,
                post_tool_max_tokens=post_tool_max_tokens,
                global_max_tokens=_post_tool_global_max(max_tokens),
                stream_callback=provider_stream_func,
                stream_message_uuid=stream_message_uuid,
                initial_response=response,
            )
            if retry_response.tool_call:
                response_tool_call = retry_response.tool_call or {}
                response = retry_response
                response_text = visibility.strip_tool_markup(
                    visibility.strip_thinking_blocks(retry_response.text)
                )
            else:
                recovered = await _last_resort_evidence_answer(
                    llm,
                    conversation,
                    max_tokens,
                    stream_callback=provider_stream_func,
                    stream_message_uuid=stream_message_uuid,
                )
                if recovered:
                    if _compact_summary_fallback_enabled():
                        response_from_compact_summary = True
                    response_text = recovered
    if is_post_tool_result_turn and (not response_tool_call):
        retrieval_notice = document_retrieval_notice(last_message)
        if retrieval_notice:
            response_text = append_retrieval_notice_if_missing(
                response_text, retrieval_notice
            )
    if enforce_citations and (not response_tool_call):
        is_streaming_turn = callable(provider_stream_func)
        require_cited_numeric_claims = current_stage() == "direct_synthesis"
        original_response_text = (response_text or "").strip()
        citations_valid = citations.response_has_required_citations(
            response_text,
            citation_allowlist,
            require_cited_numeric_claims=require_cited_numeric_claims,
        )
        original_invalid = citations.find_invalid_citations(
            original_response_text, citation_allowlist
        )
        original_has_any_citation = bool(
            citations.extract_citations(original_response_text)
        )
        should_soft_accept_original = (
            (not citations_valid)
            and (not original_invalid)
            and original_has_any_citation
            and fb.is_high_quality_summary(original_response_text)
        )
        should_soft_accept_image_display = (
            (not citations_valid)
            and (not original_invalid)
            and _latest_user_requested_image(conversation)
            and bool(imgctx.recent_tool_image_markdown(conversation, max_images=1))
        )
        should_soft_accept_compact_summary = (
            (not citations_valid)
            and response_from_compact_summary
            and (not original_invalid)
            and bool(original_response_text)
        )
        # Direct synthesis must repair uncited claims before deferred delivery;
        # fluent prose, figures, and compact summaries are not citation evidence.
        if current_stage() != "direct_synthesis" and (
            should_soft_accept_original
            or should_soft_accept_image_display
            or should_soft_accept_compact_summary
        ):
            citations_valid = True
        if (not citations_valid) and (not is_streaming_turn):
            invalid = citations.find_invalid_citations(
                response_text, citation_allowlist
            )
            prior_for_retry = response_text
            if len(prior_for_retry) > citation_retry_prior_max_chars:
                prior_for_retry = (
                    prior_for_retry[:citation_retry_prior_max_chars].rstrip()
                    + "\n[Truncated for citation retry.]"
                )
            retry_prompt = citations.build_citation_retry_prompt(
                prior_answer=prior_for_retry,
                allowed_citations=citation_allowlist,
                invalid_citations=invalid,
            )
            if require_cited_numeric_claims:
                retry_prompt += (
                    "\n\nNumeric factual sentences need citations at the claim. "
                    "For a computed comparison, cite every source supplying "
                    "an input "
                    "at that comparison; a trailing Sources list is not sufficient."
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
                "_synthesis_phase": "citation_repair",
                "system": request_system_prompt,
                "messages": retry_messages,
                "messages_pydantic": retry_pydantic_messages,
                "max_tokens": _resolve_post_tool_max_tokens(
                    conversation,
                    default_cap=post_tool_max_tokens,
                    global_max=_post_tool_global_max(max_tokens),
                ),
                "stream_callback": provider_stream_func,
                "stream_message_uuid": stream_message_uuid,
            }
            retry_response = await llm.get_message(**retry_args)
            if retry_response and (not retry_response.tool_call):
                response = retry_response
                response_text = (retry_response.text or "").strip()
                response_tool_call = {}
        retry_invalid = citations.find_invalid_citations(
            response_text, citation_allowlist
        )
        if retry_invalid and (
            not citations.response_has_required_citations(
                response_text,
                citation_allowlist,
                require_cited_numeric_claims=require_cited_numeric_claims,
            )
        ):
            if is_streaming_turn:
                response_text = (
                    response_text.rstrip()
                    + "\n\n[Note: Some citation tokens could not be verified "
                    "against retrieved chunks.]"
                )
            else:
                if _extractive_evidence_ui_enabled():
                    cited_extract = citations.synthesize_cited_extract_from_results(
                        conversation
                    ) or citations.synthesize_doc_level_extract_from_results(
                        conversation
                    )
                    if cited_extract:
                        response_text = cited_extract
                        response_tool_call = {}
        if (
            _extractive_evidence_ui_enabled()
            and (not response_tool_call)
            and (
                not response_text.strip()
                or response_text.strip()
                == visibility.clean_response_failure_text(after_tool_result=True)
            )
        ):
            cited_extract = citations.synthesize_cited_extract_from_results(
                conversation
            ) or citations.synthesize_doc_level_extract_from_results(conversation)
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
    ):
        markdown_images = imgctx.recent_tool_image_markdown(conversation, max_images=3)
        if markdown_images:
            existing_image_refs = _collect_doc_refs_from_embedded_images(response_text)
            missing_markdown_images: list[str] = []
            for line in markdown_images:
                image_ref = _extract_doc_ref_from_image_url(line)
                if image_ref and image_ref in existing_image_refs:
                    continue
                missing_markdown_images.append(line)
            if missing_markdown_images:
                response_text = (
                    response_text.rstrip() + "\n\n" + "\n".join(missing_markdown_images)
                )
    should_append_final_sources = bool(
        (use_live_citation_stream or defer_stream_until_final)
        and source_allowlist
        and citations.citation_sources_append_enabled()
    )
    if (
        should_append_final_sources
        and (not response_tool_call)
        and visibility.should_append_citation_sources(response_text)
    ):
        response_text = _append_citation_sources_if_missing(
            response_text, source_allowlist
        )
    if response_tool_call:
        response_text = visibility.strip_tool_markup(
            visibility.strip_thinking_blocks(response_text)
        )
        if visibility.is_interim_assistant_text(response_text):
            response_text = ""
    else:
        response_text = visibility.sanitize_completed_response(response_text, response)
        if not response_text.strip():
            response_text = visibility.clean_response_failure_text(
                after_tool_result=is_post_tool_result_turn
            )
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
    if (
        defer_stream_until_final
        and callable(stream_func)
        and not new_msg.tool_call_id
        and new_msg.content.strip()
    ):
        await final_stream.send_final_answer_stream(
            stream_func,
            message_uuid=str(new_msg.message_uuid),
            content=new_msg.content,
            stop_reason=new_msg.stop_reason,
            usage=new_msg.usage,
        )

    return conversation + [new_msg], "changed"


__all__ = ["DIRECT_SYNTHESIS_GROUNDING", "complete_conversation_turn"]
