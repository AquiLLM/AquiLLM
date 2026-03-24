"""call_tool must pass model JSON as **kwargs; empty dict is not 'no arguments'."""
from __future__ import annotations

from django.test import SimpleTestCase

from aquillm.llm import AssistantMessage, ToolChoice, llm_tool

from apps.chat.tests.chat_message_test_support import _FakeLLMInterface


@llm_tool(
    for_whom="assistant",
    required=["a", "b"],
    param_descs={"a": "first", "b": "second"},
)
def _two_arg_tool(a: str, b: str) -> dict:
    """Test tool requiring two string parameters."""
    return {"result": f"{a}:{b}"}


class CallToolArgumentsTests(SimpleTestCase):
    def test_none_arguments_get_clear_error_not_bare_call(self):
        llm = _FakeLLMInterface([])
        msg = AssistantMessage(
            content="",
            stop_reason="tool_use",
            tool_call_id="t1",
            tool_call_name="_two_arg_tool",
            tool_call_input=None,
            tools=[_two_arg_tool],
            tool_choice=ToolChoice(type="auto"),
        )
        out = llm.call_tool(msg)
        self.assertIn("without a JSON argument object", out.result_dict.get("exception", ""))

    def test_empty_dict_runs_through_validate_not_bare_call(self):
        llm = _FakeLLMInterface([])
        msg = AssistantMessage(
            content="",
            stop_reason="tool_use",
            tool_call_id="t2",
            tool_call_name="_two_arg_tool",
            tool_call_input={},
            tools=[_two_arg_tool],
            tool_choice=ToolChoice(type="auto"),
        )
        out = llm.call_tool(msg)
        exc = out.result_dict.get("exception", "")
        self.assertTrue(exc)
        el = exc.lower()
        self.assertTrue("validation error" in el or "missing" in el or "required" in el)

    def test_valid_arguments_succeed(self):
        llm = _FakeLLMInterface([])
        msg = AssistantMessage(
            content="",
            stop_reason="tool_use",
            tool_call_id="t3",
            tool_call_name="_two_arg_tool",
            tool_call_input={"a": "1", "b": "2"},
            tools=[_two_arg_tool],
            tool_choice=ToolChoice(type="auto"),
        )
        out = llm.call_tool(msg)
        self.assertEqual(out.result_dict.get("result"), "1:2")
