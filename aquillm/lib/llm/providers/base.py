"""Base LLM interface class."""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from os import getenv
from json import dumps
from typing import Any, Awaitable, Callable, Literal, Optional

from pydantic import validate_call

from ..types.conversation import Conversation
from ..types.messages import AssistantMessage, LLM_Message, ToolMessage, UserMessage
from ..types.response import LLMResponse
from ..types.tools import LLMTool, dump_tool_choice
from . import fallback_heuristics as fb
from . import image_context as imgctx
from .summary import generate_compact_tool_summary

try:
    from aquillm.settings import DEBUG
except ImportError:
    DEBUG = False

if DEBUG:
    from pprint import pp


class LLMInterface(ABC):
    """Abstract base class for LLM provider interfaces."""

    tool_executor = ThreadPoolExecutor(max_workers=10)
    base_args: dict = {}
    client: Any = None

    @abstractmethod
    def __init__(self, client: Any):
        pass

    @abstractmethod
    async def get_message(self, *args, **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    async def token_count(self, conversation: Conversation, new_message: Optional[str] = None) -> int:
        pass

    async def _continue_cutoff_response(
        self,
        *,
        system_prompt: str,
        message_dicts: list[dict],
        messages_for_bot: list[LLM_Message],
        partial_text: str,
        max_tokens: int,
    ) -> Optional[LLMResponse]:
        if not partial_text.strip():
            return None
        continuation_prompt = (
            "Continue the previous assistant response exactly where it stopped. "
            "Do not restart, do not repeat prior points, and keep the same structure/tone."
        )
        continuation_messages = message_dicts + [
            {"role": "assistant", "content": partial_text},
            {"role": "user", "content": continuation_prompt},
        ]
        continuation_messages_pydantic: list[LLM_Message] = messages_for_bot + [
            AssistantMessage(content=partial_text, stop_reason="max_tokens"),
            UserMessage(content=continuation_prompt),
        ]
        try:
            return await self.get_message(
                **(
                    self.base_args
                    | {
                        "system": system_prompt,
                        "messages": continuation_messages,
                        "messages_pydantic": continuation_messages_pydantic,
                        "max_tokens": max_tokens,
                    }
                )
            )
        except Exception:
            return None

    def call_tool(self, message: AssistantMessage) -> ToolMessage:
        """Execute a tool call from an assistant message."""
        tools = message.tools
        if tools:
            name = message.tool_call_name
            input = message.tool_call_input
            tools_dict = {tool.llm_definition["name"]: tool for tool in tools}
            tool_name = name or "invalid_tool"
            for_whom: Literal["assistant", "user"] = "assistant"
            result_dict: dict = {"exception": "Tool call failed before execution"}
            if not name or name not in tools_dict.keys():
                result_dict = {"exception": "Function name is not valid"}
                result = imgctx.serialize_tool_result_for_llm(result_dict)
            else:
                tool = tools_dict[name]
                tool_name = tool.name
                for_whom = tool.for_whom
                if input:
                    future = self.tool_executor.submit(partial(tool, **input))
                else:
                    future = self.tool_executor.submit(tool)
                try:
                    tool_timeout_s = float(getenv("TOOL_CALL_TIMEOUT_SECONDS", "10"))
                    result_dict = future.result(timeout=tool_timeout_s)
                    result = imgctx.serialize_tool_result_for_llm(result_dict)
                except TimeoutError:
                    result_dict = {"exception": "Tool call timed out"}
                    result = imgctx.serialize_tool_result_for_llm(result_dict)
                except Exception as e:
                    if DEBUG:
                        raise
                    result_dict = {"exception": str(e)}
                    result = imgctx.serialize_tool_result_for_llm(result_dict)
            return ToolMessage(
                tool_name=tool_name,
                content=result,
                arguments=input,
                result_dict=result_dict,
                for_whom=for_whom,
                tools=message.tools,
                files=result_dict.get("files") if isinstance(result_dict, dict) else None,
                tool_choice=message.tool_choice,
            )
        raise ValueError("call_tool called on a message with no tools!")

    @validate_call
    async def complete(
        self,
        conversation: Conversation,
        max_tokens: int,
        stream_func: Optional[Callable[[dict], Awaitable[Any]]] = None,
    ) -> tuple[Conversation, Literal["changed", "unchanged"]]:
        """Complete a conversation by getting the next message from the LLM."""
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
                new_tool_msg = self.call_tool(last_message)
                return conversation + [new_tool_msg], "changed"
            return conversation, "unchanged"

        assert isinstance(last_message, (UserMessage, ToolMessage)), "Type assertion failed"
        is_post_tool_result_turn = (
            isinstance(last_message, ToolMessage) and last_message.for_whom == "assistant"
        )
        try:
            tool_step_max_tokens = max(128, int(getenv("LLM_TOOL_STEP_MAX_TOKENS", "512")))
        except Exception:
            tool_step_max_tokens = 512
        try:
            post_tool_max_tokens = max(256, int(getenv("LLM_POST_TOOL_MAX_TOKENS", "1024")))
        except Exception:
            post_tool_max_tokens = 1024
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
                self.base_args
                | tools
                | {
                    "system": system_prompt,
                    "messages": message_dicts,
                    "messages_pydantic": messages_for_bot,
                    "max_tokens": request_max_tokens,
                    "stream_callback": stream_func,
                    "stream_message_uuid": stream_message_uuid,
                }
            )
        }

        response = await self.get_message(**sdk_args)
        should_force_tool_retry = (
            bool(last_message.tools)
            and bool(last_message.tool_choice)
            and last_message.tool_choice.type == "auto"
            and not response.tool_call
            and fb.looks_like_deferred_tool_intent(response.text)
        )
        if should_force_tool_retry:
            retry_args = sdk_args | {"tool_choice": {"type": "any"}}
            response = await self.get_message(**retry_args)

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
                finalize_args = self.base_args | {
                    "system": system_prompt,
                    "messages": finalize_messages,
                    "messages_pydantic": finalize_pydantic_messages,
                    "max_tokens": min(max_tokens, post_tool_max_tokens),
                    "stream_callback": stream_func,
                    "stream_message_uuid": stream_message_uuid,
                }
                response = await self.get_message(**finalize_args)
            post_finalize_text = (response.text or "").strip()
            if (not response.tool_call) and (
                (not post_finalize_text)
                or fb.looks_like_deferred_tool_intent(post_finalize_text)
            ):
                compact_summary = await generate_compact_tool_summary(self, conversation, max_tokens)
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
                    compact_summary = await generate_compact_tool_summary(self, conversation, max_tokens)
                    response_text = compact_summary or (
                        fb.synthesize_from_recent_tool_results(conversation)
                        if fb.extractive_fallback_enabled()
                        else None
                    ) or (
                        "I completed retrieval but received an unusable tool-call payload. "
                        "Please retry and I will provide a full summary."
                    )

        if (not response_tool_call) and (not response_text.strip()):
            compact_summary = await generate_compact_tool_summary(self, conversation, max_tokens)
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
            continuation_response: Optional[LLMResponse] = None
            continuation_text = ""
            if fb.continue_on_cutoff_enabled():
                continuation_response = await self._continue_cutoff_response(
                    system_prompt=system_prompt,
                    message_dicts=message_dicts,
                    messages_for_bot=messages_for_bot,
                    partial_text=response_text,
                    max_tokens=max_tokens,
                )
                continuation_text = (
                    (continuation_response.text or "").strip() if continuation_response else ""
                )
            if continuation_text and not fb.looks_like_deferred_tool_intent(continuation_text):
                separator = "\n" if response_text and not response_text.endswith(("\n", " ")) else ""
                response_text = f"{response_text.rstrip()}{separator}{continuation_text}"
                response = continuation_response
            else:
                compact_summary = await generate_compact_tool_summary(self, conversation, max_tokens)
                if compact_summary:
                    response_text = compact_summary
                elif fb.extractive_fallback_enabled():
                    synthesized = fb.synthesize_from_recent_tool_results(conversation)
                    if synthesized:
                        response_text = synthesized
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

    async def spin(
        self,
        convo: Conversation,
        max_func_calls: int,
        send_func: Callable[[Conversation], Any],
        max_tokens: int,
        stream_func: Optional[Callable[[dict], Awaitable[Any]]] = None,
    ) -> None:
        """Spin the conversation, executing tool calls until complete."""
        calls = 0
        repeated_tool_call_count = 0
        prev_tool_signature: Optional[str] = None
        tool_name_call_counts: dict[str, int] = {}
        try:
            repeat_break_threshold = max(1, int(getenv("LLM_REPEAT_TOOL_BREAK_THRESHOLD", "3")))
        except Exception:
            repeat_break_threshold = 3
        try:
            max_calls_per_tool_name = max(1, int(getenv("LLM_MAX_CALLS_PER_TOOL_NAME", "2")))
        except Exception:
            max_calls_per_tool_name = 2
        while calls < max_func_calls:
            convo, changed = await self.complete(convo, max_tokens, stream_func=stream_func)
            await send_func(convo)
            if changed == "unchanged":
                return
            last_message = convo[-1]
            if isinstance(last_message, AssistantMessage) and last_message.tool_call_id:
                sig = f"{last_message.tool_call_name}|{dumps(last_message.tool_call_input or {}, sort_keys=True, ensure_ascii=True)}"
                if sig == prev_tool_signature:
                    repeated_tool_call_count += 1
                else:
                    repeated_tool_call_count = 0
                prev_tool_signature = sig
                tool_name = str(last_message.tool_call_name or "").strip().lower()
                if tool_name:
                    tool_name_call_counts[tool_name] = tool_name_call_counts.get(tool_name, 0) + 1
                    if tool_name_call_counts[tool_name] >= max_calls_per_tool_name:
                        break
                if repeated_tool_call_count >= repeat_break_threshold:
                    break
                calls += 1
        last = convo[-1]
        if isinstance(last, AssistantMessage) and last.tool_call_id:
            convo, _ = await self.complete(convo, max_tokens, stream_func=stream_func)
            await send_func(convo)
            convo[-1].tools = None
            convo[-1].tool_choice = None
            convo, _ = await self.complete(convo, max_tokens, stream_func=stream_func)
            await send_func(convo)


__all__ = ["LLMInterface"]
