"""OpenAI LLM interface."""
from typing import Optional, Any, override
from os import getenv
import re
import uuid

from tiktoken import encoding_for_model

from ..types.messages import AssistantMessage
from ..types.conversation import Conversation
from ..types.response import LLMResponse
from .base import LLMInterface
from .openai_streaming import consume_streaming_completion
from .openai_overflow import (
    retry_args_for_context_overflow,
    retry_args_for_timeout,
    strip_images_from_messages,
)
from .openai_tokens import (
    context_reserve_tokens,
    env_float,
    env_int,
    estimate_prompt_tokens,
    preflight_trim_for_context,
    trim_messages_for_overflow,
)
from .openai_tool_text import decode_json_dict, extract_tool_call_from_text
from .openai_tools_format import transform_openai_tool_choice, transform_openai_tools
from lib.llm.optimizations.lm_lingua2_adapter import maybe_compress_openai_style_messages


try:
    from aquillm.settings import DEBUG
except ImportError:
    DEBUG = False

if DEBUG:
    from pprint import pp


gpt_enc = encoding_for_model('gpt-4o')


class OpenAIInterface(LLMInterface):
    """LLM interface for OpenAI models."""

    @override
    def __init__(self, openai_client, model: str):
        self.client = openai_client
        self.base_args = {'model': model}

    @staticmethod
    def _trim_messages_for_overflow(arguments: dict, overflow_tokens: int) -> bool:
        return trim_messages_for_overflow(arguments, overflow_tokens)

    @classmethod
    def _estimate_prompt_tokens(cls, messages: list[dict]) -> int:
        return estimate_prompt_tokens(messages, gpt_enc)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        return env_int(name, default)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        return env_float(name, default)

    @classmethod
    def _context_reserve_tokens(cls, context_limit: int) -> tuple[int, int]:
        return context_reserve_tokens(context_limit)

    @classmethod
    def _preflight_trim_for_context(
        cls, arguments: dict, context_limit: int, extra_prompt_slack: int = 0
    ) -> None:
        preflight_trim_for_context(cls, arguments, context_limit, extra_prompt_slack)

    @staticmethod
    def _strip_images_from_messages(arguments: dict) -> bool:
        return strip_images_from_messages(arguments)

    @staticmethod
    def _retry_args_for_context_overflow(arguments: dict, exc: Exception) -> Optional[dict]:
        return retry_args_for_context_overflow(arguments, exc)

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        timeout_type_names = {
            "APITimeoutError",
            "ReadTimeout",
            "TimeoutException",
            "ConnectTimeout",
        }
        seen_ids: set[int] = set()
        current: Optional[BaseException] = exc
        while current and id(current) not in seen_ids:
            seen_ids.add(id(current))
            if current.__class__.__name__ in timeout_type_names:
                return True
            message = str(current).lower()
            if "request timed out" in message or "read timeout" in message or "timed out" in message:
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _retry_args_for_timeout(arguments: dict, attempt: int) -> Optional[dict]:
        return retry_args_for_timeout(arguments, attempt)

    @override
    async def get_message(self, *args, **kwargs) -> LLMResponse:
        kwargs.pop('messages_pydantic', None)
        kwargs.pop('thinking_budget', None)
        stream_callback = kwargs.pop('stream_callback', None)
        stream_message_uuid = str(kwargs.pop('stream_message_uuid', None) or uuid.uuid4())
        system_text = kwargs.pop('system')
        message_list = kwargs.pop('messages')
        max_tokens = kwargs.pop('max_tokens')
        tool_choice_raw = kwargs.pop('tool_choice', None)
        raw_tools = kwargs.get('tools')

        maybe_compress_openai_style_messages(message_list)

        if "[User preferences and background]" in system_text or "[Historical conversation context]" in system_text:
            system_text = (
                "You have access to retrieved user memory in the system context below. "
                "When relevant memory is present, use it directly. "
                "Do not claim you cannot remember past conversations when memory items are provided.\n\n"
                + system_text
            )

        configured_role = getenv("OPENAI_SYSTEM_ROLE", "").strip().lower()
        base_url = str(getattr(self.client, "base_url", "") or "").lower()
        is_local_compatible_endpoint = any(token in base_url for token in ("ollama", "vllm", "11434", "8000"))
        if configured_role in ("system", "developer"):
            system_role = configured_role
        else:
            system_role = "system" if is_local_compatible_endpoint else "developer"

        arguments = {
            "model": self.base_args['model'],
            "messages": [{"role": system_role, "content": system_text}] + message_list,
            "max_tokens": max_tokens,
        }
        context_limit_raw = (
            (getenv("OPENAI_CONTEXT_LIMIT", "") or "").strip()
            or (getenv("VLLM_MAX_MODEL_LEN", "") or "").strip()
        )
        try:
            context_limit = int(context_limit_raw)
        except Exception:
            context_limit = 0
        try:
            from lib.llm.utils.prompt_budget import (
                context_packer_enabled,
                maybe_pack_message_dicts_for_context,
                prompt_budget_context_limit,
                prompt_budget_max_tokens_cap,
            )

            pack_limit = context_limit if context_limit > 0 else prompt_budget_context_limit()
            if pack_limit > 0 and context_packer_enabled():
                sys_row = arguments["messages"][0]
                tail = arguments["messages"][1:]
                mt0 = min(max(int(arguments["max_tokens"]), 1), prompt_budget_max_tokens_cap())
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
                prompt_slack = self._env_int("OPENAI_COMPAT_PROMPT_SLACK_TOKENS", 256)
            else:
                prompt_slack = self._env_int("OPENAI_API_PROMPT_SLACK_TOKENS", 384)
            self._preflight_trim_for_context(arguments, context_limit, prompt_slack)
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

        if 'tools' in kwargs:
            arguments["tools"] = await transform_openai_tools(
                kwargs.pop('tools'),
                include_strict=not is_local_compatible_endpoint,
            )
            transformed_tool_choice = transform_openai_tool_choice(tool_choice_raw)
            if transformed_tool_choice is not None:
                arguments["tool_choice"] = transformed_tool_choice

        request_timeout_s = float(getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "120"))
        try:
            max_request_timeout_s = float(getenv("OPENAI_REQUEST_TIMEOUT_MAX_SECONDS", "360"))
        except Exception:
            max_request_timeout_s = 360.0
        if max_request_timeout_s < request_timeout_s:
            max_request_timeout_s = request_timeout_s
        try:
            max_overflow_retries = int(getenv("OPENAI_CONTEXT_OVERFLOW_RETRIES", "3"))
        except Exception:
            max_overflow_retries = 3
        if max_overflow_retries < 3:
            max_overflow_retries = 3
        try:
            max_timeout_retries = int(getenv("OPENAI_TIMEOUT_RETRIES", "2"))
        except Exception:
            max_timeout_retries = 2
        if max_timeout_retries < 0:
            max_timeout_retries = 0

        stream_enabled = (
            callable(stream_callback)
            and getenv("OPENAI_STREAM_RESPONSES", "1").strip().lower() in ("1", "true", "yes", "on")
        )
        parsed_response: Optional[LLMResponse] = None
        request_args = dict(arguments)
        if stream_enabled:
            request_args["stream"] = True
            request_args.setdefault("stream_options", {"include_usage": True})
        timeout_retries_used = 0
        max_total_retries = max_overflow_retries + max_timeout_retries
        for attempt in range(max_total_retries + 1):
            try:
                if stream_enabled:
                    stream = await self.client.chat.completions.create(timeout=request_timeout_s, **request_args)
                    parsed_response = await consume_streaming_completion(
                        stream=stream,
                        stream_callback=stream_callback,
                        stream_message_uuid=stream_message_uuid,
                        raw_tools=raw_tools,
                        model_name=self.base_args["model"],
                    )
                else:
                    response = await self.client.chat.completions.create(timeout=request_timeout_s, **request_args)
                    if DEBUG:
                        print("OpenAI SDK Response:")
                        pp(response)
                    text = response.choices[0].message.content
                    tool_call_payload: Optional[dict] = None
                    raw_tool_call = response.choices[0].message.tool_calls[0] if response.choices[0].message.tool_calls else None
                    if raw_tool_call:
                        tool_call_payload = {
                            "tool_call_id": raw_tool_call.id or str(uuid.uuid4()),
                            "tool_call_name": raw_tool_call.function.name,
                            "tool_call_input": decode_json_dict(raw_tool_call.function.arguments),
                        }
                    elif text and raw_tools:
                        tool_call_payload = extract_tool_call_from_text(text, raw_tools)
                        if tool_call_payload and re.fullmatch(
                            r"\s*(```[\s\S]*```|<function_call>[\s\S]*</function_call>|<tool_call>[\s\S]*</tool_call>|<\w+>\s*\{[\s\S]*\}\s*</\w+>)\s*",
                            text,
                            flags=re.IGNORECASE,
                        ):
                            text = None

                    if tool_call_payload and tool_call_payload.get("tool_call_name") == 'message_to_user':
                        parsed_args = tool_call_payload.get("tool_call_input") or {}
                        text = parsed_args.get('message') or text
                        tool_call_payload = None

                    parsed_response = LLMResponse(
                        text=text,
                        tool_call=tool_call_payload or {},
                        stop_reason=response.choices[0].finish_reason,
                        input_usage=response.usage.prompt_tokens,
                        output_usage=response.usage.completion_tokens,
                        model=self.base_args['model'],
                        message_uuid=stream_message_uuid,
                    )
                break
            except Exception as exc:
                retry_args = self._retry_args_for_context_overflow(request_args, exc)
                if retry_args is not None and attempt < max_total_retries:
                    request_args = retry_args
                    continue

                if (
                    self._is_timeout_error(exc)
                    and timeout_retries_used < max_timeout_retries
                    and attempt < max_total_retries
                ):
                    timeout_retries_used += 1
                    timeout_retry_args = self._retry_args_for_timeout(
                        request_args,
                        timeout_retries_used,
                    )
                    if timeout_retry_args is not None:
                        request_args = timeout_retry_args
                    request_timeout_s = min(
                        max_request_timeout_s,
                        max(request_timeout_s, request_timeout_s * 1.5),
                    )
                    continue

                if attempt >= max_total_retries:
                    raise
                raise
        assert parsed_response is not None
        return parsed_response
                        
    @override 
    async def token_count(self, conversation: Conversation, new_message: Optional[str] = None) -> int:
        assistant_messages = [message for message in conversation if isinstance(message, AssistantMessage)]
        if assistant_messages:
            return assistant_messages[-1].usage + (len(gpt_enc.encode(new_message)) if new_message else 0)
        return len(gpt_enc.encode(new_message)) if new_message else 0


__all__ = ['OpenAIInterface', 'gpt_enc']
