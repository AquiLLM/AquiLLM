"""Single-turn conversation completion orchestration for LLMInterface."""
from __future__ import annotations

import uuid
import re
from os import getenv
from typing import Any, Awaitable, Callable, Literal, Optional

from ..types.conversation import Conversation
from ..types.messages import AssistantMessage, LLM_Message, ToolMessage, UserMessage
from ..types.response import LLMResponse
from ..types.tools import ToolChoice, dump_tool_choice
from . import fallback_heuristics as fb
from . import image_context as imgctx
from . import rag_citations as citations
from . import visibility
from .retrieval_status import append_retrieval_notice_if_missing, document_retrieval_notice
from .summary import generate_compact_tool_summary

try:
    from aquillm.settings import DEBUG
except ImportError:
    DEBUG = False

if DEBUG:
    from pprint import pp


_DOC_IMAGE_URL_RE = re.compile(r"/aquillm/document_image/([^/]+)/")
def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(getenv(name, str(default)))
    except Exception:
        value = default
    return max(minimum, value)


def _conversation_used_whole_document(conversation: Conversation) -> bool:
    for msg in reversed(conversation.messages):
        if isinstance(msg, ToolMessage) and msg.for_whom == "assistant":
            if msg.tool_name in {"whole_document", "search_single_document"}:
                return True
    return False


def _post_tool_output_ceiling() -> int:
    return _env_int("LLM_POST_TOOL_OUTPUT_MAX_TOKENS", 6144, minimum=256)


def _post_tool_global_max(global_max: int) -> int:
    return max(global_max, _post_tool_output_ceiling())


def _resolve_post_tool_max_tokens(
    conversation: Conversation,
    *,
    default_cap: int,
    global_max: int,
) -> int:
    cap = max(default_cap, _post_tool_output_ceiling())
    if _conversation_used_whole_document(conversation):
        cap = max(
            cap,
            _env_int("LLM_POST_TOOL_WHOLE_DOC_MAX_TOKENS", 6144, minimum=256),
        )
    return min(cap, max(global_max, _post_tool_output_ceiling()))


def _resolve_continuation_max_tokens(
    conversation: Conversation,
    *,
    default_cap: int,
    post_tool_budget: int,
    global_max: int,
) -> int:
    cap = default_cap
    if _conversation_used_whole_document(conversation):
        cap = max(
            cap,
            _env_int("LLM_CONTINUATION_WHOLE_DOC_MAX_TOKENS", 2048, minimum=128),
        )
    cap = max(cap, post_tool_budget // 2)
    return min(global_max, post_tool_budget, cap)


def _compact_summary_fallback_enabled() -> bool:
    return getenv("LLM_ALLOW_COMPACT_SUMMARY_FALLBACK", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _extractive_evidence_ui_enabled() -> bool:
    """When false, never replace a failed synthesis with raw chunk/doc bullet dumps."""
    return getenv("LLM_ALLOW_EXTRACTIVE_EVIDENCE_UI", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _post_tool_evidence_retry_enabled() -> bool:
    return getenv("LLM_POST_TOOL_ALLOW_EVIDENCE_RETRY", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _post_tool_synthesis_retry_count() -> int:
    return min(4, _env_int("LLM_POST_TOOL_SYNTHESIS_RETRIES", 2, minimum=0))


def _latest_user_turn(conversation: Conversation) -> UserMessage | None:
    for msg in reversed(conversation.messages):
        if isinstance(msg, UserMessage):
            return msg
    return None


def _post_tool_synthesis_unsatisfied(text: Optional[str]) -> bool:
    visible = visibility.strip_tool_markup(visibility.strip_thinking_blocks(text)).strip()
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
    user_turn = _latest_user_turn(conversation)
    query = (user_turn.content or "").strip() if user_turn else ""
    wants_figures = _latest_user_requested_image(conversation)
    lines = [
        f"User request: {query or 'Answer using the retrieved documents above.'}",
        "",
        "Write a complete, thorough final answer using evidence already in this conversation.",
        "Use multiple sections when helpful; aim for depth (typically 400-900 words when sources support it).",
        "Answer directly in plain text; do not describe future retrieval steps or emit tool markup.",
    ]
    if wants_figures:
        lines.append(
            "Include relevant figures using markdown image syntax from tool results "
            "(![caption](url))."
        )
    lines.append("Explain equations and technical terms in readable language.")
    if attempt >= 1:
        lines.append(
            "Your previous reply was empty or incomplete. Synthesize a thorough answer "
            "from the document text and tool results above."
        )
    if attempt >= 2:
        lines.append(
            "Cover the main thesis, methods or math (with intuition), and key figures or findings."
        )
    if attempt >= 3 and _post_tool_evidence_retry_enabled():
        lines.append(
            "If existing excerpts are too thin for a specific claim, you may call "
            "vector_search or search_single_document once with a focused query, then stop."
        )
    return "\n".join(lines)


async def _run_post_tool_synthesis_attempt(
    llm: Any,
    *,
    system_prompt: str,
    message_dicts: list[dict],
    messages_for_bot: list[LLM_Message],
    conversation: Conversation,
    post_tool_max_tokens: int,
    global_max_tokens: int,
    stream_callback: Optional[Callable[[dict], Awaitable[Any]]],
    stream_message_uuid: str,
    attempt: int,
    allow_tools: bool,
) -> LLMResponse:
    prompt = _build_synthesis_retry_prompt(conversation, attempt)
    retry_messages = message_dicts + [{"role": "user", "content": prompt}]
    retry_pydantic_messages = messages_for_bot + [UserMessage(content=prompt)]
    retry_args: dict[str, Any] = llm.base_args | {
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


async def _retry_post_tool_synthesis(
    llm: Any,
    conversation: Conversation,
    *,
    system_prompt: str,
    message_dicts: list[dict],
    messages_for_bot: list[LLM_Message],
    post_tool_max_tokens: int,
    global_max_tokens: int,
    stream_callback: Optional[Callable[[dict], Awaitable[Any]]],
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
    stream_callback: Optional[Callable[[dict], Awaitable[Any]]] = None,
    stream_message_uuid: Optional[str] = None,
) -> Optional[str]:
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


def _doc_ref(doc_id: Any) -> str | None:
    if doc_id is None:
        return None
    doc_text = str(doc_id).strip()
    if not doc_text:
        return None
    return f"[doc:{doc_text}]"


def _extract_doc_ref_from_image_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = _DOC_IMAGE_URL_RE.search(value)
    if not match:
        return None
    return _doc_ref(match.group(1))


def _collect_doc_refs_from_embedded_images(text: str) -> set[str]:
    refs: set[str] = set()
    for match in _DOC_IMAGE_URL_RE.finditer(text or ""):
        ref = _doc_ref(match.group(1))
        if ref:
            refs.add(ref)
    return refs


def _latest_user_requested_image(conversation: Conversation) -> bool:
    for msg in reversed(conversation.messages):
        if isinstance(msg, UserMessage):
            return imgctx.looks_like_image_display_request(msg.content)
    return False


def _collect_source_refs_from_tool_message(tool_message: ToolMessage) -> set[str]:
    refs: set[str] = set()

    if isinstance(tool_message.arguments, dict):
        direct_doc_ref = _doc_ref(tool_message.arguments.get("doc_id"))
        if direct_doc_ref:
            refs.add(direct_doc_ref)

    result_dict = tool_message.result_dict if isinstance(tool_message.result_dict, dict) else {}
    payload = result_dict.get("result")
    payload_rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        payload_rows = [payload]
    elif isinstance(payload, list):
        payload_rows = [row for row in payload if isinstance(row, dict)]

    for row in payload_rows[:40]:
        row_doc_ref = _doc_ref(row.get("doc_id") or row.get("d"))
        if row_doc_ref:
            refs.add(row_doc_ref)
        image_url_ref = _extract_doc_ref_from_image_url(row.get("image_url") or row.get("u"))
        if image_url_ref:
            refs.add(image_url_ref)

    if not refs and isinstance(payload, str):
        image_url_ref = _extract_doc_ref_from_image_url(payload)
        if image_url_ref:
            refs.add(image_url_ref)

    return refs


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


def _continuation_separator(partial_text: str, continuation_text: str) -> str:
    if not partial_text:
        return ""
    if partial_text.endswith(("\n", " ")):
        return ""
    if imgctx.has_unterminated_markdown_image(partial_text):
        return ""
    if continuation_text.startswith((")", "]", "/", ".", ",", ":", ";", "!", "?")):
        return ""
    return "\n"


def _largest_suffix_prefix_overlap(left: str, right: str, min_chars: int = 1) -> int:
    max_size = min(len(left), len(right))
    for size in range(max_size, max(min_chars, 1) - 1, -1):
        if left.endswith(right[:size]):
            return size
    return 0


def _trim_duplicate_continuation_prefix(partial_text: str, continuation_text: str) -> str:
    partial = partial_text or ""
    continuation = continuation_text or ""
    if (not partial) or (not continuation):
        return continuation

    trimmed = False
    overlap_floor = max(16, min(96, min(len(partial), len(continuation)) // 3))
    suffix_overlap = _largest_suffix_prefix_overlap(partial, continuation, min_chars=overlap_floor)
    if suffix_overlap:
        continuation = continuation[suffix_overlap:]
        trimmed = True

    if trimmed:
        continuation = continuation.lstrip("\r\n")
    return continuation


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
        request_system_prompt = (
            f"{request_system_prompt}\n\n"
            "Final synthesis step: answer the user using retrieved document content already in "
            "this thread. Write a thorough, well-structured user-facing answer with enough detail "
            "to stand alone; finish every section and do not stop mid-sentence. Do not emit status "
            "lines, tool markup, or promises to retrieve later."
        )
    source_allowlist = set(citation_allowlist)
    if is_post_tool_result_turn and not source_allowlist:
        source_allowlist = _collect_source_refs_from_tool_message(last_message)
    use_live_citation_stream = bool(
        source_allowlist and callable(stream_func) and citations.citation_sources_append_enabled()
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
                out["content"] = _append_citation_sources_if_missing(content, source_allowlist)
            await stream_func(out)

        effective_stream_func = _live_citation_stream
    tool_step_max_tokens = _env_int("LLM_TOOL_STEP_MAX_TOKENS", 512, minimum=128)
    post_tool_max_tokens = _env_int("LLM_POST_TOOL_MAX_TOKENS", 3072, minimum=256)
    continuation_max_tokens = _env_int("LLM_CONTINUATION_MAX_TOKENS", 1536, minimum=128)
    citation_retry_prior_max_chars = _env_int("LLM_CITATION_RETRY_PRIOR_MAX_CHARS", 2400, minimum=512)
    request_max_tokens = max_tokens
    if isinstance(last_message, UserMessage) and last_message.tools:
        request_max_tokens = min(max_tokens, tool_step_max_tokens)
    elif is_post_tool_result_turn:
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
    tool_choice_type = str(getattr(last_message.tool_choice, "type", "") or "").strip().lower()
    should_force_tool_retry = (
        bool(last_message.tools)
        and bool(last_message.tool_choice)
        and tool_choice_type in {"auto", "any"}
        and not response.tool_call
        and fb.looks_like_deferred_tool_intent(response.text)
    )
    if should_force_tool_retry:
        retry_args = sdk_args | {"tool_choice": {"type": "any"}}
        response = await llm.get_message(**retry_args)

    if is_post_tool_result_turn and not response.tool_call:
        response = await _retry_post_tool_synthesis(
            llm,
            conversation,
            system_prompt=request_system_prompt,
            message_dicts=message_dicts,
            messages_for_bot=messages_for_bot,
            post_tool_max_tokens=post_tool_max_tokens,
            global_max_tokens=_post_tool_global_max(max_tokens),
            stream_callback=effective_stream_func,
            stream_message_uuid=stream_message_uuid,
            initial_response=response,
        )

    allowed_tool_names = {tool.name for tool in (last_message.tools or [])}
    response_text = visibility.strip_tool_markup(visibility.strip_thinking_blocks(response.text))
    response_tool_call = response.tool_call or {}

    if response_tool_call:
        called_tool_name = response_tool_call.get("tool_call_name")
        if (not allowed_tool_names) or (called_tool_name not in allowed_tool_names):
            response_tool_call = {}
            if not response_text.strip():
                recovered = await _last_resort_evidence_answer(
                    llm,
                    conversation,
                    max_tokens,
                    stream_callback=effective_stream_func,
                    stream_message_uuid=stream_message_uuid,
                )
                response_text = recovered or (
                    "I completed retrieval but received an unusable tool-call payload. "
                    "Please retry and I will provide a full summary."
                )

    if (not response_tool_call) and visibility.is_interim_assistant_text(response_text):
        response_text = ""

    if (not response_tool_call) and (not response_text.strip()):
        recovered = await _last_resort_evidence_answer(
            llm,
            conversation,
            max_tokens,
            stream_callback=effective_stream_func,
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
        continuation_response: Optional[LLMResponse] = None
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
                stream_callback=effective_stream_func,
                stream_message_uuid=stream_message_uuid,
            )
            if continuation_response is not None and not continuation_response.message_uuid:
                continuation_response.message_uuid = stream_message_uuid
            raw_continuation_text = (
                (continuation_response.text or "").strip() if continuation_response else ""
            )
            continuation_text = _trim_duplicate_continuation_prefix(response_text, raw_continuation_text)
            if raw_continuation_text and (not continuation_text):
                preserve_partial_response = True
        if continuation_text and not fb.looks_like_deferred_tool_intent(continuation_text):
            separator = _continuation_separator(response_text, continuation_text)
            response_text = f"{response_text.rstrip()}{separator}{continuation_text}"
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
                stream_callback=effective_stream_func,
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
                    stream_callback=effective_stream_func,
                    stream_message_uuid=stream_message_uuid,
                )
                if recovered:
                    if _compact_summary_fallback_enabled():
                        response_from_compact_summary = True
                    response_text = recovered
    if is_post_tool_result_turn and (not response_tool_call):
        retrieval_notice = document_retrieval_notice(last_message)
        if retrieval_notice:
            response_text = append_retrieval_notice_if_missing(response_text, retrieval_notice)
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
        if (
            should_soft_accept_original
            or should_soft_accept_image_display
            or should_soft_accept_compact_summary
        ):
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
                "max_tokens": _resolve_post_tool_max_tokens(
                    conversation,
                    default_cap=post_tool_max_tokens,
                    global_max=_post_tool_global_max(max_tokens),
                ),
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
                if _extractive_evidence_ui_enabled():
                    cited_extract = (
                        citations.synthesize_cited_extract_from_results(conversation)
                        or citations.synthesize_doc_level_extract_from_results(conversation)
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
            cited_extract = (
                citations.synthesize_cited_extract_from_results(conversation)
                or citations.synthesize_doc_level_extract_from_results(conversation)
            )
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
                response_text = response_text.rstrip() + "\n\n" + "\n".join(missing_markdown_images)
    if (
        use_live_citation_stream
        and (not response_tool_call)
        and visibility.should_append_citation_sources(response_text)
    ):
        response_text = _append_citation_sources_if_missing(response_text, source_allowlist)
    if response_tool_call:
        response_text = visibility.strip_tool_markup(visibility.strip_thinking_blocks(response_text))
        if visibility.is_interim_assistant_text(response_text):
            response_text = ""
    else:
        response_text = visibility.sanitize_assistant_text(response_text)
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

    return conversation + [new_msg], "changed"


__all__ = ["complete_conversation_turn"]
