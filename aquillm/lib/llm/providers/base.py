"""Base LLM interface class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from json import dumps
from typing import Any, Literal

import structlog
from pydantic import validate_call

from ..types.conversation import Conversation
from ..types.messages import AssistantMessage, LLM_Message, ToolMessage, UserMessage
from ..types.response import LLMResponse
from .complete_turn import complete_conversation_turn
from .tool_budget import ToolBudgetConfig, ToolBudgetPolicy, ToolCallObservation

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
    async def token_count(
        self, conversation: Conversation, new_message: str | None = None
    ) -> int:
        pass

    async def _continue_cutoff_response(
        self,
        *,
        system_prompt: str,
        message_dicts: list[dict],
        messages_for_bot: list[LLM_Message],
        partial_text: str,
        max_tokens: int,
        stream_message_uuid: str | None = None,
        stream_callback: Callable[[dict], Awaitable[Any]] | None = None,
    ) -> LLMResponse | None:
        if not partial_text.strip():
            return None
        continuation_prompt = (
            "Continue the previous assistant response exactly where "
            "it stopped. "
            "Do not restart from the beginning, do not repeat prior "
            "points or section headings, "
            "and keep the same structure/tone. "
            "If the previous text ended mid-sentence, mid-heading, "
            "or mid-list item, "
            "complete that fragment first before starting any new section."
        )
        continuation_messages = message_dicts + [
            {"role": "assistant", "content": partial_text},
            {"role": "user", "content": continuation_prompt},
        ]
        continuation_messages_pydantic: list[LLM_Message] = messages_for_bot + [
            AssistantMessage(content=partial_text, stop_reason="max_tokens"),
            UserMessage(content=continuation_prompt),
        ]
        continuation_stream_callback = stream_callback
        if callable(stream_callback):

            async def _prepend_partial_to_stream(payload: dict) -> Any:
                out = dict(payload)
                content = str(out.get("content", ""))
                out["content"] = f"{partial_text}{content}"
                await stream_callback(out)

            continuation_stream_callback = _prepend_partial_to_stream
        try:
            return await self.get_message(
                **(
                    self.base_args
                    | {
                        "system": system_prompt,
                        "messages": continuation_messages,
                        "messages_pydantic": continuation_messages_pydantic,
                        "max_tokens": max_tokens,
                        # The first pass already performed synthesis reasoning.
                        # A cutoff continuation is a mechanical completion of
                        # that answer, so another full reasoning pass only adds
                        # latency and can restart the response.
                        "thinking_budget": 0,
                        "_synthesis_phase": "continuation",
                        "stream_callback": continuation_stream_callback,
                        "stream_message_uuid": stream_message_uuid,
                    }
                )
            )
        except Exception:
            return None

    def call_tool(self, message: AssistantMessage) -> ToolMessage:
        from .tool_execution import call_tool

        return call_tool(self, message)

    @validate_call
    async def complete(
        self,
        conversation: Conversation,
        max_tokens: int,
        stream_func: Callable[[dict], Awaitable[Any]] | None = None,
    ) -> tuple[Conversation, Literal["changed", "unchanged"]]:
        """Complete a conversation by getting the next message from the LLM."""
        from lib.llm.turn_context import (
            check_turn_active,
            evidence_handoff,
            fenced_callback,
        )

        check_turn_active()
        stream_func = fenced_callback(stream_func)
        handoff = await evidence_handoff(self, conversation, max_tokens, stream_func)
        if handoff is not None:
            return handoff
        result = await complete_conversation_turn(
            self, conversation, max_tokens, stream_func=stream_func
        )
        check_turn_active()
        return result

    async def spin(
        self,
        convo: Conversation,
        max_func_calls: int,
        send_func: Callable[[Conversation], Any],
        max_tokens: int,
        stream_func: Callable[[dict], Awaitable[Any]] | None = None,
    ) -> None:
        """Spin the conversation, executing tool calls until complete."""
        calls = 0
        stop_reason: str | None = None
        cached_tool_result: ToolMessage | None = None
        budget_policy = ToolBudgetPolicy(
            ToolBudgetConfig.from_env(max_func_calls=max_func_calls)
        )
        from lib.llm.turn_context import (
            current_turn,
            fenced_callback,
        )

        state = current_turn()
        if state:
            state.policy = budget_policy
            state.max_func_calls = min(
                state.max_func_calls or max_func_calls, max_func_calls
            )
        send_func = fenced_callback(send_func)
        while calls < max_func_calls:
            convo, changed = await self.complete(
                convo, max_tokens, stream_func=stream_func
            )
            await send_func(convo)
            if changed == "unchanged":
                logger.info(
                    "obs.llm.tool_loop_ended",
                    stop_reason=stop_reason or "unchanged",
                    calls=calls,
                    policy=budget_policy.summary(),
                )
                return
            last_message = convo[-1]
            if isinstance(last_message, AssistantMessage) and last_message.tool_call_id:
                arguments = dumps(
                    last_message.tool_call_input or {},
                    sort_keys=True,
                    ensure_ascii=True,
                )
                sig = f"{last_message.tool_call_name}|{arguments}"
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
                    cached_tool_result = _matching_tool_result(convo, last_message)
                    break
                calls += 1
        if stop_reason is None and calls >= max_func_calls:
            stop_reason = "max_func_calls_reached"
        logger.info(
            "obs.llm.tool_loop_ended",
            stop_reason=stop_reason or "loop_exited",
            calls=calls,
            policy=budget_policy.summary(),
        )
        last = convo[-1]
        if isinstance(last, AssistantMessage) and last.tool_call_id:
            if cached_tool_result is not None:
                reused_result = cached_tool_result.model_copy(
                    update={
                        "content": (
                            "This identical tool call was already completed; reuse the "
                            "preceding result."
                        ),
                        "arguments": dict(last.tool_call_input or {}),
                        "result_dict": dict(cached_tool_result.result_dict or {}),
                    }
                )
                convo = convo + [reused_result]
                await send_func(convo)
            else:
                convo, _ = await self.complete(
                    convo,
                    max_tokens,
                    stream_func=stream_func,
                )
                await send_func(convo)
            convo[-1].tools = None
            convo[-1].tool_choice = None
            convo, _ = await self.complete(convo, max_tokens, stream_func=stream_func)
            await send_func(convo)


def _latest_tool_result_for_name(convo: Conversation, tool_name: str) -> dict | None:
    if not tool_name:
        return None
    for message in reversed(list(convo)):
        if (
            isinstance(message, ToolMessage)
            and message.for_whom == "assistant"
            and str(message.tool_name or "").strip().lower() == tool_name
        ):
            return (
                message.result_dict if isinstance(message.result_dict, dict) else None
            )
    return None


def _matching_tool_result(
    convo: Conversation,
    pending: AssistantMessage,
) -> ToolMessage | None:
    tool_name = str(pending.tool_call_name or "").strip().lower()
    if not tool_name:
        return None
    pending_arguments = dumps(
        pending.tool_call_input or {},
        sort_keys=True,
        ensure_ascii=True,
    )
    for message in reversed(list(convo)[:-1]):
        if not isinstance(message, ToolMessage) or message.for_whom != "assistant":
            continue
        if str(message.tool_name or "").strip().lower() != tool_name:
            continue
        message_arguments = dumps(
            message.arguments or {},
            sort_keys=True,
            ensure_ascii=True,
        )
        if message_arguments == pending_arguments:
            return message
    return None


__all__ = ["LLMInterface"]
