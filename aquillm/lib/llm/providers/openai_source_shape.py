"""Shared OpenAI request framing and a pure protected text-only capability."""

from os import getenv

_ABSENT = object()


def base_request_arguments(provider, system_text, message_list, max_tokens, thinking_budget):
    if ("[User preferences and background]" in system_text
            or "[Historical conversation context]" in system_text):
        system_text = (
            "You have access to retrieved user memory in the system context below. "
            "When relevant memory is present, use it directly. "
            "Do not claim you cannot remember past conversations when memory items are provided. "
            "Do not describe internal memory tools, storage backends, or persistence mechanisms. "
            "If the user asks you to remember something, acknowledge it naturally without discussing whether "
            "a tool is available or whether storage will happen behind the scenes.\n\n" + system_text
        )
    configured_role = getenv("OPENAI_SYSTEM_ROLE", "").strip().lower()
    base_url = str(getattr(provider.client, "base_url", "") or "").lower()
    local = any(token in base_url for token in ("ollama", "vllm", "11434", "8000"))
    role = configured_role if configured_role in ("system", "developer") else (
        "system" if local else "developer")
    arguments = {"model": provider.base_args["model"],
        "messages": [{"role": role, "content": system_text}] + message_list,
        "max_tokens": max_tokens}
    thinking = requested_thinking(thinking_budget, arguments)
    if "vllm" in base_url or "8000" in base_url:
        thinking = ((getenv("OPENAI_COMPAT_ENABLE_THINKING", "1") or "1").strip().lower()
            in ("1", "true", "yes", "on")) if thinking_budget is None else int(thinking_budget) != 0
        arguments["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}
    return arguments, thinking, local


def requested_thinking(thinking_budget, arguments):
    try:
        thinking = True if thinking_budget is None else int(thinking_budget) != 0
    except Exception:
        thinking = True
    return arguments.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking", thinking)


def configured_sampling(arguments):
    from .openai_runtime_config import optional_float

    for key, name in (("temperature", "OPENAI_TEMPERATURE"), ("top_p", "OPENAI_TOP_P")):
        value = optional_float(name)
        if value is not None:
            arguments[key] = value


def protected_text_request_arguments(provider, *, system_text, message_list,
        max_tokens, thinking_budget=None, response_format=None, tools=_ABSENT,
        tool_choice=None, stream=False):
    """None means this exact optional preflight shape is not supported."""
    if (type(system_text) is not str or type(message_list) is not list
            or not message_list or type(max_tokens) is not int or max_tokens <= 0
            or tools is not _ABSENT or tool_choice is not None or stream is not False
            or any(type(row) is not dict or set(row) != {"role", "content"}
                or row["role"] not in {"system", "developer", "user", "assistant"}
                or type(row["content"]) is not str for row in message_list)):
        return None
    arguments, _, _ = base_request_arguments(
        provider, system_text, message_list, max_tokens, thinking_budget)
    configured_sampling(arguments)
    if response_format is not None:
        arguments["response_format"] = response_format
    return arguments
