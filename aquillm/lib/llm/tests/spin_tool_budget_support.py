"""Integration tests for adaptive tool budget behavior in LLMInterface.spin()."""
from __future__ import annotations

from lib.llm.providers.base import LLMInterface
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage


class _DummyLLMInterface(LLMInterface):
    def __init__(self, client=None):
        self.client = client
        self.base_args = {}

    async def get_message(self, *args, **kwargs):
        raise NotImplementedError

    async def token_count(self, conversation: Conversation, new_message: str | None = None) -> int:
        return 0


def _assistant_tool_call(call_number: int, query: str = "alpha") -> AssistantMessage:
    return AssistantMessage(
        content=f"tool call {call_number}",
        stop_reason="tool_use",
        tool_call_id=f"tc-{call_number}",
        tool_call_name="vector_search",
        tool_call_input={"q": query},
    )


def _assistant_final_answer(text: str = "final synthesis answer") -> AssistantMessage:
    return AssistantMessage(content=text, stop_reason="stop")


def _tool_result(
    call_number: int,
    *,
    exception: bool = False,
    query: str = "alpha",
) -> ToolMessage:
    result_dict = {"exception": "Tool call timed out"} if exception else {"result": [{"id": call_number}]}
    return ToolMessage(
        content=str(result_dict),
        tool_name="vector_search",
        for_whom="assistant",
        arguments={"q": query},
        result_dict=result_dict,
    )


def _convo_with_n_calls(
    n_calls: int,
    *,
    same_query: bool = True,
    exception_results: bool = False,
) -> Conversation:
    messages: list = [UserMessage(content="Find relevant sources.")]
    for i in range(1, n_calls + 1):
        if i > 1:
            prior_query = "same-query" if same_query else f"query-{i - 1}"
            messages.append(
                _tool_result(
                    i - 1,
                    exception=exception_results,
                    query=prior_query,
                )
            )
        query = "same-query" if same_query else f"query-{i}"
        messages.append(_assistant_tool_call(i, query=query))
    return Conversation(system="sys", messages=messages)
