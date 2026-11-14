"""Prepare OpenAI-compatible requests and fit them to the context budget."""

import asyncio
from dataclasses import dataclass

from lib.llm.evidence_guard import current_protection, validate_request

from .openai_tools_format import transform_openai_tool_choice, transform_openai_tools
from .openai_source_shape import (
    base_request_arguments,
    configured_sampling,
    protected_text_request_arguments,
    requested_thinking,
)


@dataclass(frozen=True)
class PreparedRequest:
    arguments: dict
    thinking_requested: bool


async def prepare_request(
    provider,
    *,
    system_text,
    message_list,
    max_tokens,
    thinking_budget,
    tool_choice_raw,
    kwargs,
    compress_messages,
) -> PreparedRequest:
    protection = current_protection()
    if (protection is not None and not protection.allow_history_trim
            and "tools" not in kwargs and tool_choice_raw is None
            and not ("response_format" in kwargs and kwargs["response_format"] is None)):
        arguments = protected_text_request_arguments(provider,
            system_text=system_text, message_list=message_list, max_tokens=max_tokens,
            thinking_budget=thinking_budget, response_format=kwargs.get("response_format"))
        if arguments is not None:
            validate_request(arguments, output_reserve=arguments["max_tokens"])
            return PreparedRequest(arguments, requested_thinking(thinking_budget, arguments))
    arguments, thinking_requested, is_local_compatible_endpoint = base_request_arguments(
        provider, system_text, message_list, max_tokens, thinking_budget)
    from .openai_runtime_config import context_limit as configured_limit

    context_limit = configured_limit()
    should_compress = True
    protection = current_protection()
    if context_limit > 0 and protection is None:
        prompt_tokens = provider._estimate_prompt_tokens(arguments["messages"])
        available_prompt_tokens = max(1, context_limit - max(0, int(max_tokens)))
        compression_trigger_tokens = max(1, int(available_prompt_tokens * 0.8))
        should_compress = prompt_tokens >= compression_trigger_tokens
    if should_compress and protection is None:
        await asyncio.to_thread(
            compress_messages,
            message_list,
        )
    try:
        from lib.llm.utils.prompt_budget import (
            cap_completion_tokens,
            context_packer_enabled,
            maybe_pack_message_dicts_for_context,
            prompt_budget_context_limit,
        )

        pack_limit = (
            context_limit if context_limit > 0 else prompt_budget_context_limit()
        )
        if pack_limit > 0 and context_packer_enabled() and protection is None:
            sys_row = arguments["messages"][0]
            tail = arguments["messages"][1:]
            mt0 = cap_completion_tokens(arguments["max_tokens"])
            _, mt1 = maybe_pack_message_dicts_for_context(
                str(sys_row.get("content", "")),
                tail,
                context_limit=pack_limit,
                max_tokens=mt0,
            )
            arguments["max_tokens"] = mt1
    except Exception:
        pass
    if context_limit > 0 and (protection is None or protection.allow_history_trim):
        if is_local_compatible_endpoint:
            prompt_slack = provider._env_int("OPENAI_COMPAT_PROMPT_SLACK_TOKENS", 256)
        else:
            prompt_slack = provider._env_int("OPENAI_API_PROMPT_SLACK_TOKENS", 384)
        provider._preflight_trim_for_context(arguments, context_limit, prompt_slack)
    configured_sampling(arguments)

    if "tools" in kwargs:
        arguments["tools"] = await transform_openai_tools(
            kwargs.pop("tools"),
            include_strict=not is_local_compatible_endpoint,
        )
        transformed_tool_choice = transform_openai_tool_choice(tool_choice_raw)
        if transformed_tool_choice is not None:
            arguments["tool_choice"] = transformed_tool_choice

    if "response_format" in kwargs:
        arguments["response_format"] = kwargs.pop("response_format")

    validate_request(arguments, output_reserve=arguments["max_tokens"])
    return PreparedRequest(arguments, thinking_requested)


def is_timeout_error(exc: Exception) -> bool:
    timeout_type_names = {
        "APITimeoutError",
        "ReadTimeout",
        "TimeoutException",
        "ConnectTimeout",
    }
    seen_ids: set[int] = set()
    current: BaseException | None = exc
    while current and id(current) not in seen_ids:
        seen_ids.add(id(current))
        if current.__class__.__name__ in timeout_type_names:
            return True
        message = str(current).lower()
        if (
            "request timed out" in message
            or "read timeout" in message
            or "timed out" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False
