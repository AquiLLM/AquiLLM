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
        messages = arguments.get("messages")
        if not isinstance(messages, list) or len(messages) <= 1:
            return False

        if len(messages) >= 3:
            del messages[1]
            return True

        candidate_indices = [
            i for i in range(1, len(messages) - 1)
            if isinstance(messages[i], dict) and isinstance(messages[i].get("content"), str)
        ]
        if not candidate_indices:
            candidate_indices = [
                i for i in range(1, len(messages))
                if isinstance(messages[i], dict) and isinstance(messages[i].get("content"), str)
            ]
        if not candidate_indices:
            return False

        idx = candidate_indices[0]
        content = str(messages[idx].get("content", ""))
        if not content:
            return False

        trim_chars = max(128, overflow_tokens * 12)
        if len(content) <= trim_chars:
            messages[idx]["content"] = "[Earlier context trimmed due to token limit.]"
        else:
            messages[idx]["content"] = content[trim_chars:]
        return True

    @staticmethod
    def _flatten_content_for_token_estimate(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            text_val = content.get("text")
            if isinstance(text_val, str):
                return text_val
            part_type = str(content.get("type", "")).lower()
            if part_type in {"image_url", "input_image", "image"}:
                image_val = content.get("image_url")
                if isinstance(image_val, dict):
                    url_val = image_val.get("url")
                    if isinstance(url_val, str):
                        return url_val
                if isinstance(image_val, str):
                    return image_val
                direct_url = content.get("url")
                if isinstance(direct_url, str):
                    return direct_url
            content_val = content.get("content")
            if isinstance(content_val, str):
                return content_val
            return str(content)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    part_type = str(item.get("type", "")).lower()
                    if part_type in {"image_url", "input_image", "image"}:
                        image_val = item.get("image_url")
                        if isinstance(image_val, dict):
                            url_val = image_val.get("url")
                            if isinstance(url_val, str):
                                parts.append(url_val)
                                continue
                        if isinstance(image_val, str):
                            parts.append(image_val)
                            continue
                        direct_url = item.get("url")
                        if isinstance(direct_url, str):
                            parts.append(direct_url)
                            continue
                    text_val = item.get("text")
                    if isinstance(text_val, str):
                        parts.append(text_val)
                        continue
                    content_val = item.get("content")
                    if isinstance(content_val, str):
                        parts.append(content_val)
            return "\n".join(parts)
        return str(content)

    @classmethod
    def _estimate_prompt_tokens(cls, messages: list[dict]) -> int:
        total = 12
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", ""))
            content = cls._flatten_content_for_token_estimate(msg.get("content", ""))
            total += 6 + len(gpt_enc.encode(role)) + len(gpt_enc.encode(content))
        return total

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int((getenv(name, str(default)) or str(default)).strip())
        except Exception:
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float((getenv(name, str(default)) or str(default)).strip())
        except Exception:
            return default

    @classmethod
    def _context_reserve_tokens(cls, context_limit: int) -> tuple[int, int]:
        mode = str(getenv("OPENAI_CONTEXT_RESERVE_MODE", "ratio") or "ratio").strip().lower()
        if mode == "fixed":
            guard_tokens = max(64, cls._env_int("OPENAI_CONTEXT_GUARD_TOKENS", 256))
            estimator_pad_tokens = max(0, cls._env_int("OPENAI_ESTIMATOR_PAD_TOKENS", 256))
            return guard_tokens, estimator_pad_tokens

        guard_ratio = min(max(cls._env_float("OPENAI_CONTEXT_GUARD_RATIO", 0.015), 0.0), 0.5)
        pad_ratio = min(max(cls._env_float("OPENAI_ESTIMATOR_PAD_RATIO", 0.0075), 0.0), 0.5)

        guard_min = max(64, cls._env_int("OPENAI_CONTEXT_GUARD_MIN_TOKENS", 96))
        guard_max = max(guard_min, cls._env_int("OPENAI_CONTEXT_GUARD_MAX_TOKENS", 4096))
        pad_min = max(0, cls._env_int("OPENAI_ESTIMATOR_PAD_MIN_TOKENS", 64))
        pad_max = max(pad_min, cls._env_int("OPENAI_ESTIMATOR_PAD_MAX_TOKENS", 2048))

        guard_tokens = int(context_limit * guard_ratio)
        estimator_pad_tokens = int(context_limit * pad_ratio)
        guard_tokens = min(guard_max, max(guard_min, guard_tokens))
        estimator_pad_tokens = min(pad_max, max(pad_min, estimator_pad_tokens))
        return guard_tokens, estimator_pad_tokens

    @classmethod
    def _preflight_trim_for_context(cls, arguments: dict, context_limit: int) -> None:
        messages = arguments.get("messages")
        if not isinstance(messages, list) or len(messages) <= 1 or context_limit <= 0:
            return

        guard_tokens, estimator_pad_tokens = cls._context_reserve_tokens(context_limit)

        has_tools = bool(arguments.get("tools"))
        min_completion_tokens = 128 if has_tools else 256
        current_max_tokens = int(arguments.get("max_tokens", 0))
        if current_max_tokens <= 0:
            return

        prompt_budget = context_limit - current_max_tokens - guard_tokens - estimator_pad_tokens
        if prompt_budget < 256:
            reduced_completion_tokens = max(min_completion_tokens, context_limit - guard_tokens - estimator_pad_tokens - 256)
            if 0 < reduced_completion_tokens < current_max_tokens:
                arguments["max_tokens"] = reduced_completion_tokens
                current_max_tokens = reduced_completion_tokens
                prompt_budget = context_limit - current_max_tokens - guard_tokens - estimator_pad_tokens

        if prompt_budget <= 0:
            return

        prompt_tokens = cls._estimate_prompt_tokens(messages)
        trim_loops = 0
        while prompt_tokens > prompt_budget and trim_loops < 32:
            overflow_estimate = max(1, prompt_tokens - prompt_budget)
            if not cls._trim_messages_for_overflow(arguments, overflow_estimate):
                break
            messages = arguments.get("messages")
            if not isinstance(messages, list):
                break
            prompt_tokens = cls._estimate_prompt_tokens(messages)
            trim_loops += 1

        if prompt_tokens > prompt_budget:
            available_completion_tokens = context_limit - prompt_tokens - guard_tokens - estimator_pad_tokens
            if min_completion_tokens <= available_completion_tokens < current_max_tokens:
                arguments["max_tokens"] = available_completion_tokens

    @staticmethod
    def _strip_images_from_messages(arguments: dict) -> bool:
        """Remove image content from messages to recover from context overflow."""
        messages = arguments.get("messages")
        if not isinstance(messages, list):
            return False
        
        stripped = False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "")
                        if part_type == "image_url" or part_type == "image":
                            stripped = True
                            new_content.append({
                                "type": "text",
                                "text": "[Image removed due to context limit]"
                            })
                        else:
                            new_content.append(part)
                    else:
                        new_content.append(part)
                if stripped:
                    text_parts = [p.get("text", "") for p in new_content if isinstance(p, dict) and p.get("type") == "text"]
                    if len(text_parts) == len(new_content):
                        msg["content"] = "\n".join(text_parts)
                    else:
                        msg["content"] = new_content
        
        return stripped

    @staticmethod
    def _retry_args_for_context_overflow(arguments: dict, exc: Exception) -> Optional[dict]:
        """Parse context overflow error and adjust arguments for retry."""
        message = str(exc)
        match = re.search(
            r"passed\s+(\d+)\s+input tokens.*maximum input length of\s+(\d+)\s+tokens",
            message,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        passed_input_tokens = int(match.group(1))
        max_input_tokens = int(match.group(2))
        overflow = passed_input_tokens - max_input_tokens
        if overflow <= 0:
            return None
        current_max_tokens = int(arguments.get("max_tokens", 0))
        has_tools = bool(arguments.get("tools"))
        try:
            if has_tools:
                min_completion_tokens = max(64, int(getenv("LLM_TOOL_MIN_COMPLETION_TOKENS", "128")))
                hard_floor_tokens = 64
            else:
                min_completion_tokens = max(128, int(getenv("LLM_MIN_COMPLETION_TOKENS", "384")))
                hard_floor_tokens = 192
        except Exception:
            min_completion_tokens = 128 if has_tools else 384
            hard_floor_tokens = 64 if has_tools else 192

        retry_args = dict(arguments)
        if isinstance(arguments.get("messages"), list):
            retry_args["messages"] = [
                dict(msg) if isinstance(msg, dict) else msg
                for msg in arguments["messages"]
            ]

        changed = False

        # Data-URL images can dominate prompt size. Strip them on any overflow retry.
        if OpenAIInterface._strip_images_from_messages(retry_args):
            changed = True
            return retry_args

        if current_max_tokens > min_completion_tokens:
            if overflow <= 4:
                safety_margin = 8
            else:
                safety_margin = max(32, min(192, overflow * 4))
            reduced_max_tokens = max(min_completion_tokens, current_max_tokens - overflow - safety_margin)
            if reduced_max_tokens >= current_max_tokens:
                reduced_max_tokens = max(min_completion_tokens, current_max_tokens - 1)
            if reduced_max_tokens != current_max_tokens:
                retry_args["max_tokens"] = reduced_max_tokens
                changed = True

        if not changed and current_max_tokens > hard_floor_tokens:
            emergency_margin = max(16, min(128, overflow * 4))
            emergency_reduced_max_tokens = max(
                hard_floor_tokens,
                current_max_tokens - overflow - emergency_margin,
            )
            if emergency_reduced_max_tokens >= current_max_tokens:
                emergency_reduced_max_tokens = max(
                    hard_floor_tokens,
                    current_max_tokens - 1,
                )
            if emergency_reduced_max_tokens < current_max_tokens:
                retry_args["max_tokens"] = emergency_reduced_max_tokens
                changed = True

        should_trim_context = (overflow > 16) or (current_max_tokens <= (min_completion_tokens + 16))
        if should_trim_context and OpenAIInterface._trim_messages_for_overflow(retry_args, overflow):
            changed = True
        
        if not changed and OpenAIInterface._strip_images_from_messages(retry_args):
            changed = True

        return retry_args if changed else None

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
        current_max_tokens = int(arguments.get("max_tokens", 0))
        if current_max_tokens <= 0:
            return None
        has_tools = bool(arguments.get("tools"))
        try:
            min_completion_tokens = max(
                64 if has_tools else 128,
                int(getenv("LLM_TOOL_MIN_COMPLETION_TOKENS", "128"))
                if has_tools else int(getenv("LLM_MIN_COMPLETION_TOKENS", "256")),
            )
        except Exception:
            min_completion_tokens = 128 if has_tools else 256

        retry_args = dict(arguments)
        if isinstance(arguments.get("messages"), list):
            retry_args["messages"] = [
                dict(msg) if isinstance(msg, dict) else msg
                for msg in arguments["messages"]
            ]

        changed = False
        reduction_ratio = min(0.6, 0.2 + (0.1 * attempt))
        reduction_tokens = max(64, int(current_max_tokens * reduction_ratio))
        reduced_max_tokens = max(min_completion_tokens, current_max_tokens - reduction_tokens)
        if reduced_max_tokens < current_max_tokens:
            retry_args["max_tokens"] = reduced_max_tokens
            changed = True

        if attempt >= 2 and OpenAIInterface._trim_messages_for_overflow(
            retry_args,
            overflow_tokens=256 * (attempt - 1),
        ):
            changed = True

        return retry_args if changed else None

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
                self._preflight_trim_for_context(arguments, context_limit)
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
