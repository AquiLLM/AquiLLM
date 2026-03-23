"""OpenAI LLM interface."""
from typing import Optional, Any, override
from os import getenv
from json import loads
import re
import uuid

from tiktoken import encoding_for_model

from ..types.messages import AssistantMessage
from ..types.conversation import Conversation
from ..types.response import LLMResponse
from .base import LLMInterface
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

    @staticmethod
    def _decode_json_dict(raw: Any) -> dict:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        start = None
        depth = 0
        in_string = False
        escaped = False
        for i, ch in enumerate(text):
            if start is None:
                if ch == "{":
                    start = i
                    depth = 1
                    in_string = False
                    escaped = False
                continue
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        return None

    @staticmethod
    def _tool_call_from_payload(payload: dict, allowed_tools: set[str]) -> Optional[dict]:
        if not isinstance(payload, dict):
            return None
        name = payload.get("name") or payload.get("tool_name")
        args = payload.get("arguments")
        if args is None:
            args = payload.get("args")
        if args is None:
            args = payload.get("parameters")

        if isinstance(name, str) and name in allowed_tools:
            if not isinstance(args, dict):
                args = {}
            return {
                "tool_call_id": str(uuid.uuid4()),
                "tool_call_name": name,
                "tool_call_input": args,
            }

        if len(payload) == 1:
            only_name = next(iter(payload.keys()))
            only_args = payload[only_name]
            if isinstance(only_name, str) and only_name in allowed_tools and isinstance(only_args, dict):
                return {
                    "tool_call_id": str(uuid.uuid4()),
                    "tool_call_name": only_name,
                    "tool_call_input": only_args,
                }
        return None

    def _extract_tool_call_from_text(self, text: str, raw_tools: Optional[list[dict]]) -> Optional[dict]:
        if not text or not raw_tools:
            return None
        allowed_tools = {
            tool.get("name")
            for tool in raw_tools
            if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        }
        allowed_tools.discard(None)
        if not allowed_tools:
            return None

        candidates = [text]
        candidates.extend(re.findall(r"```(?:json|xml)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE))
        candidates.extend(re.findall(r"<function_call>\s*([\s\S]*?)\s*</function_call>", text, flags=re.IGNORECASE))
        candidates.extend(re.findall(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text, flags=re.IGNORECASE))

        for candidate in candidates:
            payload = self._decode_json_dict(candidate)
            if not payload:
                json_obj = self._extract_first_json_object(candidate)
                if json_obj:
                    payload = self._decode_json_dict(json_obj)
            parsed = self._tool_call_from_payload(payload, allowed_tools)
            if parsed:
                return parsed

        for tool_name in sorted(allowed_tools, key=len, reverse=True):
            pattern = rf"<{re.escape(tool_name)}>\s*([\s\S]*?)\s*</{re.escape(tool_name)}>"
            direct_match = re.search(pattern, text, flags=re.IGNORECASE)
            if not direct_match:
                continue
            args_payload = self._decode_json_dict(direct_match.group(1))
            if not args_payload:
                json_obj = self._extract_first_json_object(direct_match.group(1))
                if json_obj:
                    args_payload = self._decode_json_dict(json_obj)
            if isinstance(args_payload, dict):
                return {
                    "tool_call_id": str(uuid.uuid4()),
                    "tool_call_name": tool_name,
                    "tool_call_input": args_payload,
                }

        return None

    async def _transform_tools(self, tools: list[dict], include_strict: bool = True) -> list[dict]:
        strict_tools = include_strict and getenv("OPENAI_TOOL_STRICT", "0").strip().lower() in ("1", "true", "yes", "on")
        transformed: list[dict] = []
        for tool in tools:
            function_payload = {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": {
                        "type": "object",
                        "properties": tool['input_schema']['properties'],
                        "required": tool['input_schema'].get('required', []),
                        "additionalProperties": False
                    },
                }
            if strict_tools:
                function_payload["strict"] = True
            transformed.append({
                "type": "function",
                "function": function_payload,
            })
        return transformed

    def _transform_tool_choice(self, tool_choice: dict | None) -> str | dict | None:
        if not tool_choice:
            return None
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "tool" and tool_choice.get("name"):
            return {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
        return None

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
        if is_local_compatible_endpoint:
            context_limit_raw = (
                (getenv("OPENAI_CONTEXT_LIMIT", "") or "").strip()
                or (getenv("VLLM_MAX_MODEL_LEN", "") or "").strip()
            )
            try:
                context_limit = int(context_limit_raw)
            except Exception:
                context_limit = 0
            if context_limit > 0:
                compat_slack = self._env_int("OPENAI_COMPAT_PROMPT_SLACK_TOKENS", 256)
                self._preflight_trim_for_context(arguments, context_limit, compat_slack)
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
            arguments["tools"] = await self._transform_tools(
                kwargs.pop('tools'),
                include_strict=not is_local_compatible_endpoint,
            )
            transformed_tool_choice = self._transform_tool_choice(tool_choice_raw)
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
                    text_parts: list[str] = []
                    tool_call_parts: dict[int, dict[str, Any]] = {}
                    finish_reason = "stop"
                    input_usage = 0
                    output_usage = 0

                    async for chunk in stream:
                        choices = getattr(chunk, "choices", None) or []
                        if choices:
                            choice = choices[0]
                            delta = getattr(choice, "delta", None)
                            if delta is not None:
                                content_piece = getattr(delta, "content", None)
                                if content_piece:
                                    piece = str(content_piece)
                                    text_parts.append(piece)
                                    await stream_callback({
                                        "message_uuid": stream_message_uuid,
                                        "role": "assistant",
                                        "content": "".join(text_parts),
                                        "done": False,
                                    })

                                for tc in (getattr(delta, "tool_calls", None) or []):
                                    idx = int(getattr(tc, "index", 0) or 0)
                                    entry = tool_call_parts.setdefault(
                                        idx,
                                        {"id": None, "name_parts": [], "arg_parts": []},
                                    )
                                    tc_id = getattr(tc, "id", None)
                                    if tc_id:
                                        entry["id"] = str(tc_id)
                                    fn = getattr(tc, "function", None)
                                    if fn is not None:
                                        fn_name = getattr(fn, "name", None)
                                        if fn_name:
                                            entry["name_parts"].append(str(fn_name))
                                        fn_args = getattr(fn, "arguments", None)
                                        if fn_args:
                                            entry["arg_parts"].append(str(fn_args))

                            finish_reason_chunk = getattr(choice, "finish_reason", None)
                            if finish_reason_chunk:
                                finish_reason = str(finish_reason_chunk)

                        usage = getattr(chunk, "usage", None)
                        if usage is not None:
                            input_usage = int(getattr(usage, "prompt_tokens", input_usage) or input_usage)
                            output_usage = int(getattr(usage, "completion_tokens", output_usage) or output_usage)

                    text = "".join(text_parts) or None
                    tool_call_payload: Optional[dict] = None
                    if tool_call_parts:
                        first_idx = sorted(tool_call_parts.keys())[0]
                        first_tool_call = tool_call_parts[first_idx]
                        tool_name = "".join(first_tool_call["name_parts"]).strip()
                        tool_args = "".join(first_tool_call["arg_parts"])
                        if tool_name:
                            tool_call_payload = {
                                "tool_call_id": first_tool_call["id"] or str(uuid.uuid4()),
                                "tool_call_name": tool_name,
                                "tool_call_input": self._decode_json_dict(tool_args),
                            }
                    elif text and raw_tools:
                        tool_call_payload = self._extract_tool_call_from_text(text, raw_tools)
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

                    await stream_callback({
                        "message_uuid": stream_message_uuid,
                        "role": "assistant",
                        "content": text or "",
                        "done": True,
                        "usage": input_usage + output_usage,
                    })

                    parsed_response = LLMResponse(
                        text=text,
                        tool_call=tool_call_payload or {},
                        stop_reason=finish_reason,
                        input_usage=input_usage,
                        output_usage=output_usage,
                        model=self.base_args['model'],
                        message_uuid=stream_message_uuid,
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
                            "tool_call_input": self._decode_json_dict(raw_tool_call.function.arguments),
                        }
                    elif text and raw_tools:
                        tool_call_payload = self._extract_tool_call_from_text(text, raw_tools)
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
