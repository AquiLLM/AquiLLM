"""Unit tests for _EchoLLMInterface."""
import uuid

from django.test import SimpleTestCase
from asgiref.sync import async_to_sync

from aquillm.llm import Conversation, UserMessage, LLMResponse

from apps.chat.tests.chat_message_test_support import _EchoLLMInterface


class EchoPlainTextTests(SimpleTestCase):
    def test_plain_text_is_echoed_back(self):
        llm = _EchoLLMInterface()
        convo = Conversation(system="test", messages=[UserMessage(content="hello world")])
        response = async_to_sync(llm.get_message)(
            messages=[m.render() for m in convo.messages],
            messages_pydantic=list(convo.messages),
        )
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.text, "hello world")
        self.assertEqual(response.stop_reason, "end_turn")
        self.assertEqual(response.tool_call, {})
        self.assertEqual(response.input_usage, 0)
        self.assertEqual(response.output_usage, 0)


class EchoJsonTextTests(SimpleTestCase):
    def test_json_text_only_returns_text_with_end_turn(self):
        llm = _EchoLLMInterface()
        convo = Conversation(
            system="test",
            messages=[UserMessage(content='{"text": "hello from json"}')],
        )
        response = async_to_sync(llm.get_message)(
            messages=[m.render() for m in convo.messages],
            messages_pydantic=list(convo.messages),
        )
        self.assertEqual(response.text, "hello from json")
        self.assertEqual(response.stop_reason, "end_turn")
        self.assertEqual(response.tool_call, {})


class EchoToolCallTests(SimpleTestCase):
    def test_json_with_tool_returns_tool_use(self):
        llm = _EchoLLMInterface()
        convo = Conversation(
            system="test",
            messages=[UserMessage(content='{"text": "searching", "tool": "vector_search", "input": {"query": "stars"}}')],
        )
        response = async_to_sync(llm.get_message)(
            messages=[m.render() for m in convo.messages],
            messages_pydantic=list(convo.messages),
        )
        self.assertEqual(response.text, "searching")
        self.assertEqual(response.stop_reason, "tool_use")
        self.assertEqual(response.tool_call["tool_call_name"], "vector_search")
        self.assertEqual(response.tool_call["tool_call_input"], {"query": "stars"})
        uuid.UUID(response.tool_call["tool_call_id"])

    def test_json_with_text_and_tool_but_no_input_returns_end_turn(self):
        llm = _EchoLLMInterface()
        convo = Conversation(
            system="test",
            messages=[UserMessage(content='{"text": "no tool", "tool": "vector_search"}')],
        )
        response = async_to_sync(llm.get_message)(
            messages=[m.render() for m in convo.messages],
            messages_pydantic=list(convo.messages),
        )
        self.assertEqual(response.text, "no tool")
        self.assertEqual(response.stop_reason, "end_turn")
        self.assertEqual(response.tool_call, {})
