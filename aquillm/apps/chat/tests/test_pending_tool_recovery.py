"""Pending tool calls must use current authorized runtime bindings after reload."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from aquillm.llm import (
    AssistantMessage,
    Conversation,
    LLMTool,
    ToolMessage,
    UserMessage,
)
from lib.llm.providers.tool_execution import call_tool


def _tool(name, result):
    return LLMTool(
        llm_definition={
            "name": name,
            "description": name,
            "input_schema": {"type": "object", "properties": {}},
        },
        for_whom="assistant",
        _function=lambda: {"result": result},
    )


def _pending_call():
    return Conversation(
        system="system",
        messages=[
            UserMessage(content="Look it up"),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="call-1",
                tool_call_name="lookup",
                tool_call_input={},
            ),
        ],
    )


def test_pending_call_rebinds_only_current_authorized_tool():
    convo = _pending_call()
    current = _tool("lookup", "fresh authorized result")

    convo.rebind_tools([current])

    assert convo[-1].tools == [current]
    assert convo[-1].tools[0]() == {"result": "fresh authorized result"}


def test_reloaded_pending_call_executes_current_tool_and_returns_result():
    convo = _pending_call()
    convo.rebind_tools([_tool("lookup", "fresh authorized result")])

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = call_tool(SimpleNamespace(tool_executor=executor), convo[-1])

    assert result.tool_name == "lookup"
    assert result.result_dict == {"result": "fresh authorized result"}


def test_unavailable_pending_call_is_settled_as_tool_failure():
    convo = _pending_call()

    convo.rebind_tools([])

    assert isinstance(convo[-1], ToolMessage)
    assert convo[-1].for_whom == "assistant"
    assert convo[-1].tool_name == "lookup"
    assert convo[-1].result_dict.get("exception")
