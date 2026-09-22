"""Prepare OpenAI-compatible requests and fit them to the context budget."""
import asyncio
from dataclasses import dataclass
from os import getenv

from .openai_tools_format import transform_openai_tool_choice, transform_openai_tools


@dataclass(frozen=True)
class PreparedRequest:
    arguments: dict
    thinking_requested: bool


async def prepare_request(
    provider, *, system_text, message_list, max_tokens, thinking_budget,
    tool_choice_raw, kwargs, compress_messages,
) -> PreparedRequest:
    if (
        "[User preferences and background]" in system_text
        or "[Historical conversation context]" in system_text
    ):
        system_text = (
            "You have access to retrieved user memory in the system context below. "
            "When relevant memory is present, use it directly. "
            "Do not claim you cannot remember past conversations when memory items are provided. "
            "Do not describe internal memory tools, storage backends, or persistence mechanisms. "
            "If the user asks you to remember something, acknowledge it naturally without discussing whether "
            "a tool is available or whether storage will happen behind the scenes.\n\n"
            + system_text
        )

    configured_role = getenv("OPENAI_SYSTEM_ROLE", "").strip().lower()
    base_url = str(getattr(provider.client, "base_url", "") or "").lower()
    is_local_compatible_endpoint = any(
        token in base_url for token in ("ollama", "vllm", "11434", "8000")
    )
    if configured_role in ("system", "developer"):
        system_role = configured_role
    else:
        system_role = "system" if is_local_compatible_endpoint else "developer"

    arguments = {
        "model": provider.base_args["model"],
        "messages": [{"role": system_role, "content": system_text}] + message_list,
        "max_tokens": max_tokens,
    }
    is_vllm_endpoint = "vllm" in base_url or "8000" in base_url
    try:
        thinking_requested = (
            True if thinking_budget is None else int(thinking_budget) != 0
        )
    except Exception:
        thinking_requested = True
    if is_vllm_endpoint:
        if thinking_budget is None:
            enable_thinking = (
                getenv("OPENAI_COMPAT_ENABLE_THINKING", "1") or "1"
            ).strip().lower() in ("1", "true", "yes", "on")
        else:
            enable_thinking = int(thinking_budget) != 0
        thinking_requested = enable_thinking
        arguments["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }
    context_limit_raw = (getenv("OPENAI_CONTEXT_LIMIT", "") or "").strip() or (
        getenv("VLLM_MAX_MODEL_LEN", "") or ""
    ).strip()
    try:
        context_limit = int(context_limit_raw)
    except Exception:
        context_limit = 0
    should_compress = True
    if context_limit > 0:
        prompt_tokens = provider._estimate_prompt_tokens(arguments["messages"])
        available_prompt_tokens = max(1, context_limit - max(0, int(max_tokens)))
        compression_trigger_tokens = max(1, int(available_prompt_tokens * 0.8))
        should_compress = prompt_tokens >= compression_trigger_tokens
    if should_compress:
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
        if pack_limit > 0 and context_packer_enabled():
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
    if context_limit > 0:
        if is_local_compatible_endpoint:
            prompt_slack = provider._env_int("OPENAI_COMPAT_PROMPT_SLACK_TOKENS", 256)
        else:
            prompt_slack = provider._env_int("OPENAI_API_PROMPT_SLACK_TOKENS", 384)
        provider._preflight_trim_for_context(arguments, context_limit, prompt_slack)
    temp_raw = (getenv("OPENAI_TEMPERATURE", "") or "").strip()
    if temp_raw:
        try:
            arguments["temperature"] = float(temp_raw)
        except Exception:
            pass
    top_p_raw = (getenv("OPENAI_TOP_P", "") or "").strip()
    if top_p_raw:
        try:
            arguments["top_p"] = float(top_p_raw)
        except Exception:
            pass

    if "tools" in kwargs:
        arguments["tools"] = await transform_openai_tools(
            kwargs.pop("tools"),
            include_strict=not is_local_compatible_endpoint,
        )
        transformed_tool_choice = transform_openai_tool_choice(tool_choice_raw)
        if transformed_tool_choice is not None:
            arguments["tool_choice"] = transformed_tool_choice

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
