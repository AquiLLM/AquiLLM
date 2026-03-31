"""Base LLM interface class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from json import dumps
from os import getenv
from typing import Any, Awaitable, Callable, Literal, Optional

from pydantic import validate_call
import structlog

from ..types.conversation import Conversation
from ..types.messages import AssistantMessage, LLM_Message, ToolMessage, UserMessage
from ..types.response import LLMResponse
from .tool_budget import ToolBudgetConfig, ToolBudgetPolicy, ToolCallObservation
from . import image_context as imgctx
from .complete_turn import complete_conversation_turn
from ..utils.tool_call_kwargs import normalize_tool_call_kwargs

try:
    from aquillm.settings import DEBUG
except ImportError:
    DEBUG = False

logger = structlog.stdlib.get_logger(__name__)


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
        stream_message_uuid: Optional[str] = None,
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
                        "stream_message_uuid": stream_message_uuid,
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
            call_arguments: Any = input
            if not name or name not in tools_dict.keys():
                result_dict = {"exception": "Function name is not valid"}
                result = imgctx.serialize_tool_result_for_llm(result_dict)
            else:
                tool = tools_dict[name]
                tool_name = tool.name
                for_whom = tool.for_whom
                # Must use a dict for kwargs. `if input:` is wrong: `{}` is falsy but still means
                # "model sent an argument object" and should be validated (not bare tool() with no params).
                if not isinstance(input, dict):
                    result_dict = {
                        "exception": (
                            "The model returned a tool call without a JSON argument object. "
                            "Required parameters were not supplied; try again or simplify the request."
                        ),
                    }
                    result = imgctx.serialize_tool_result_for_llm(result_dict)
                else:
                    call_arguments = normalize_tool_call_kwargs(name, input)
                    future = self.tool_executor.submit(partial(tool, **call_arguments))
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
                arguments=call_arguments,
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
        return await complete_conversation_turn(self, conversation, max_tokens, stream_func=stream_func)

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
        stop_reason: Optional[str] = None
        budget_policy = ToolBudgetPolicy(ToolBudgetConfig.from_env(max_func_calls=max_func_calls))
        while calls < max_func_calls:
            convo, changed = await self.complete(convo, max_tokens, stream_func=stream_func)
            await send_func(convo)
            if changed == "unchanged":
                logger.info(
                    "tool_loop_ended stop_reason=%s calls=%d policy=%s",
                    stop_reason or "unchanged",
                    calls,
                    budget_policy.summary(),
                )
                return
            last_message = convo[-1]
            if isinstance(last_message, AssistantMessage) and last_message.tool_call_id:
                sig = f"{last_message.tool_call_name}|{dumps(last_message.tool_call_input or {}, sort_keys=True, ensure_ascii=True)}"
                tool_name = str(last_message.tool_call_name or "").strip().lower()
                latest_tool_result = _latest_tool_result_for_name(convo, tool_name)
                decision = budget_policy.observe_tool_call(
                    ToolCallObservation(
                        tool_name=tool_name,
                        signature=sig,
                        latest_result_dict=latest_tool_result,
                    )
                )
                if not decision.should_continue:
                    stop_reason = decision.stop_reason
                    break
                calls += 1
        if stop_reason is None and calls >= max_func_calls:
            stop_reason = "max_func_calls_reached"
        logger.info(
            "tool_loop_ended stop_reason=%s calls=%d policy=%s",
            stop_reason or "loop_exited",
            calls,
            budget_policy.summary(),
        )
        last = convo[-1]
        if isinstance(last, AssistantMessage) and last.tool_call_id:
            convo, _ = await self.complete(convo, max_tokens, stream_func=stream_func)
            await send_func(convo)
            convo[-1].tools = None
            convo[-1].tool_choice = None
            convo, _ = await self.complete(convo, max_tokens, stream_func=stream_func)
            await send_func(convo)


def _latest_tool_result_for_name(convo: Conversation, tool_name: str) -> Optional[dict]:
    if not tool_name:
        return None
    for message in reversed(list(convo)):
        if (
            isinstance(message, ToolMessage)
            and message.for_whom == "assistant"
            and str(message.tool_name or "").strip().lower() == tool_name
        ):
            return message.result_dict if isinstance(message.result_dict, dict) else None
    return None


__all__ = ["LLMInterface"]
