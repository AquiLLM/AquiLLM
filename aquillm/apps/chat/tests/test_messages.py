"""
Unit tests for the message storage redesign.

Tests the adapter functions (Pydantic <-> Django conversion), save/load round-trips,
rating persistence, and the frontend JSON structure. These tests don't require a browser
or WebSocket connection — they test the database layer directly.
"""

from uuid import uuid4
from django.test import TestCase, SimpleTestCase
from django.contrib.auth import get_user_model
from asgiref.sync import async_to_sync
from types import SimpleNamespace
from unittest.mock import patch

from aquillm.models import WSConversation, Message
from aquillm.llm import (
    Conversation,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    LLMInterface,
    LLMResponse,
    llm_tool,
    ToolChoice,
    OpenAIInterface,
)
from aquillm.message_adapters import (
    pydantic_message_to_django,
    django_message_to_pydantic,
    load_conversation_from_db,
    save_conversation_to_db,
    build_frontend_conversation_json,
)

User = get_user_model()


@llm_tool(
    for_whom='assistant',
    required=[],
    param_descs={},
)
def _test_document_ids():
    """Test tool for LLMInterface.complete() retry behavior."""
    return {'result': 'ok'}


@llm_tool(
    for_whom='assistant',
    required=[],
    param_descs={},
)
def _test_image_result_tool():
    return {
        "result": {"status": "ok"},
        "_images": [
            {
                "result_index": 1,
                "title": "figure",
                "image_data_url": "data:image/jpeg;base64," + ("A" * 4096),
                "caption": "chart",
            }
        ],
        "_image_instruction": "Use markdown image syntax.",
    }


class _FakeLLMInterface(LLMInterface):
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def get_message(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.responses[len(self.calls) - 1]

    async def token_count(self, conversation, new_message=None):
        return 0


class _FakeTitleLLM:
    def __init__(self, text):
        self.base_args = {}
        self._text = text

    async def get_message(self, *args, **kwargs):
        return SimpleNamespace(text=self._text)


class ToolUseRetryTests(SimpleTestCase):
    def test_retries_with_required_tool_when_model_only_promises_to_search(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text="I'll search each of these three papers for detailed information.",
                tool_call={},
                stop_reason='end_turn',
                input_usage=5,
                output_usage=5,
            ),
            LLMResponse(
                text=None,
                tool_call={
                    'tool_call_id': 'tool_1',
                    'tool_call_name': '_test_document_ids',
                    'tool_call_input': {},
                },
                stop_reason='tool_use',
                input_usage=5,
                output_usage=5,
            ),
        ])
        convo = Conversation(
            system='You are a test assistant.',
            messages=[
                UserMessage(
                    content='Search these papers and summarize memory systems.',
                    tools=[_test_document_ids],
                    tool_choice=ToolChoice(type='auto'),
                )
            ],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 512)

        self.assertEqual(changed, 'changed')
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0]['tool_choice']['type'], 'auto')
        self.assertEqual(llm.calls[1]['tool_choice']['type'], 'any')
        self.assertEqual(updated[-1].tool_call_name, '_test_document_ids')

    def test_does_not_retry_for_normal_non_tool_text_reply(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text='Here is a concise answer from prior context.',
                tool_call={},
                stop_reason='end_turn',
                input_usage=5,
                output_usage=5,
            ),
        ])
        convo = Conversation(
            system='You are a test assistant.',
            messages=[
                UserMessage(
                    content='Give me a quick answer.',
                    tools=[_test_document_ids],
                    tool_choice=ToolChoice(type='auto'),
                )
            ],
        )

        updated, _ = async_to_sync(llm.complete)(convo, 512)

        self.assertEqual(len(llm.calls), 1)
        self.assertIsNone(updated[-1].tool_call_id)


class OpenAIFallbackToolParsingTests(SimpleTestCase):
    class _FakeCompletions:
        def __init__(self, response):
            self.response = response

        async def create(self, **kwargs):
            return self.response

    class _FakeOpenAIClient:
        def __init__(self, response):
            self.chat = SimpleNamespace(
                completions=OpenAIFallbackToolParsingTests._FakeCompletions(response)
            )

    def test_extracts_tool_call_from_xml_text_when_tool_calls_missing(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[],
                        content=(
                            '<function_call>{"name":"vector_search","arguments":'
                            '{"search_string":"memory and agents","top_k":5}}</function_call>'
                        ),
                    ),
                    finish_reason='stop',
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )

        llm = OpenAIInterface(self._FakeOpenAIClient(response), model='qwen3.5:27b')
        result = async_to_sync(llm.get_message)(
            system='test system',
            messages=[{'role': 'user', 'content': 'search for memory papers'}],
            max_tokens=256,
            tools=[{
                'name': 'vector_search',
                'description': 'Search docs',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'search_string': {'type': 'string', 'description': 'query'},
                        'top_k': {'type': 'integer', 'description': 'k'},
                    },
                    'required': ['search_string', 'top_k'],
                },
            }],
            tool_choice={'type': 'auto'},
        )

        self.assertEqual(result.tool_call['tool_call_name'], 'vector_search')
        self.assertEqual(result.tool_call['tool_call_input']['search_string'], 'memory and agents')
        self.assertEqual(result.tool_call['tool_call_input']['top_k'], 5)
        self.assertIsNone(result.text)


class OpenAIContextOverflowRetryTests(SimpleTestCase):
    class _SequencedCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise Exception(
                    "Error code: 400 - {'error': {'message': "
                    "\"You passed 31745 input tokens and requested 1024 output tokens. "
                    "However, the model's context length is only 32768 tokens, resulting in "
                    "a maximum input length of 31744 tokens. Please reduce the length of the "
                    "input prompt. (parameter=input_tokens, value=31745)\"}}"
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(tool_calls=[], content='Recovered after retry'),
                        finish_reason='stop',
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
            )

    class _FakeOpenAIClient:
        def __init__(self, completions):
            self.chat = SimpleNamespace(completions=completions)

    def test_retries_once_with_lower_max_tokens_on_context_overflow(self):
        completions = self._SequencedCompletions()
        llm = OpenAIInterface(self._FakeOpenAIClient(completions), model='qwen3.5:27b')

        result = async_to_sync(llm.get_message)(
            system='test system',
            messages=[{'role': 'user', 'content': 'hello'}],
            max_tokens=1024,
        )

        self.assertEqual(result.text, 'Recovered after retry')
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(completions.calls[0]['max_tokens'], 1024)
        self.assertLess(completions.calls[1]['max_tokens'], 1024)
        self.assertGreaterEqual(completions.calls[1]['max_tokens'], 64)

    def test_one_token_overflow_retry_trims_messages_not_only_max_tokens(self):
        """vLLM can reject when 1 token over; retries must trim, not only lower max_tokens."""
        user_content = "word " * 800
        arguments = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 476,
            "tools": [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}],
        }
        exc = Exception(
            "You passed 32293 input tokens and requested 476 output tokens. "
            "However, the model's context length is only 32768 tokens, resulting in "
            "a maximum input length of 32292 tokens."
        )
        retry = OpenAIInterface._retry_args_for_context_overflow(dict(arguments), exc)
        self.assertIsNotNone(retry)
        self.assertLess(retry["max_tokens"], 476)
        self.assertNotEqual(retry["messages"][1].get("content"), user_content)

    class _ImageSensitiveCompletions:
        def __init__(self):
            self.calls = []

        @staticmethod
        def _has_image_content(messages) -> bool:
            if not isinstance(messages, list):
                return False
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in {"image_url", "image", "input_image"}:
                        return True
            return False

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise Exception(
                    "Error code: 400 - {'error': {'message': "
                    "\"You passed 31745 input tokens and requested 1024 output tokens. "
                    "However, the model's context length is only 32768 tokens, resulting in "
                    "a maximum input length of 31744 tokens. Please reduce the length of the "
                    "input prompt. (parameter=input_tokens, value=31745)\"}}"
                )

            if self._has_image_content(kwargs.get("messages")):
                raise Exception(
                    "Error code: 400 - {'error': {'message': "
                    "\"You passed 31745 input tokens and requested 1015 output tokens. "
                    "However, the model's context length is only 32768 tokens, resulting in "
                    "a maximum input length of 31753 tokens. Please reduce the length of the "
                    "input prompt. (parameter=input_tokens, value=31745)\"}}"
                )

            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(tool_calls=[], content='Recovered after image stripping retry'),
                        finish_reason='stop',
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=18),
            )

    def test_retries_with_images_stripped_even_for_small_overflow(self):
        completions = self._ImageSensitiveCompletions()
        llm = OpenAIInterface(self._FakeOpenAIClient(completions), model='qwen3.5:27b')

        result = async_to_sync(llm.get_message)(
            system='test system',
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Analyze this chart.'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + ('A' * 8000)}},
                ],
            }],
            max_tokens=1024,
        )

        self.assertEqual(result.text, 'Recovered after image stripping retry')
        self.assertEqual(len(completions.calls), 2)
        second_message_content = completions.calls[1]['messages'][1]['content']
        self.assertIsInstance(second_message_content, str)
        self.assertIn("Image removed due to context limit", second_message_content)


class OpenAIContextReserveScalingTests(SimpleTestCase):
    def test_ratio_reserve_scales_with_context_limit(self):
        small_guard, small_pad = OpenAIInterface._context_reserve_tokens(12_288)
        large_guard, large_pad = OpenAIInterface._context_reserve_tokens(100_000)

        self.assertGreaterEqual(small_guard, 64)
        self.assertGreaterEqual(small_pad, 0)
        self.assertGreater(large_guard, small_guard)
        self.assertGreater(large_pad, small_pad)

    def test_preflight_trim_uses_ratio_reserve_by_default(self):
        arguments = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "x"},
            ],
            "max_tokens": 2048,
        }
        with patch.object(OpenAIInterface, "_estimate_prompt_tokens", return_value=9700), patch.object(
            OpenAIInterface,
            "_trim_messages_for_overflow",
            return_value=True,
        ) as trim_mock:
            OpenAIInterface._preflight_trim_for_context(arguments, context_limit=12_288)

        trim_mock.assert_not_called()

    def test_token_estimator_counts_image_url_payload(self):
        text_only = [{
            "role": "user",
            "content": [{"type": "text", "text": "caption"}],
        }]
        with_image = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "caption"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + ("A" * 4096)}},
            ],
        }]

        text_tokens = OpenAIInterface._estimate_prompt_tokens(text_only)
        image_tokens = OpenAIInterface._estimate_prompt_tokens(with_image)

        self.assertGreater(image_tokens, text_tokens)

    @patch.dict(
        "os.environ",
        {
            "OPENAI_CONTEXT_RESERVE_MODE": "fixed",
            "OPENAI_CONTEXT_GUARD_TOKENS": "512",
            "OPENAI_ESTIMATOR_PAD_TOKENS": "256",
        },
        clear=False,
    )
    def test_fixed_mode_keeps_legacy_behavior(self):
        arguments = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "x"},
            ],
            "max_tokens": 2048,
        }
        with patch.object(OpenAIInterface, "_estimate_prompt_tokens", return_value=9700), patch.object(
            OpenAIInterface,
            "_trim_messages_for_overflow",
            return_value=True,
        ) as trim_mock:
            OpenAIInterface._preflight_trim_for_context(arguments, context_limit=12_288)

        trim_mock.assert_called()


class ToolMessageSafetyTests(SimpleTestCase):
    def test_render_uses_text_only_and_redacts_data_urls(self):
        msg = ToolMessage(
            content="{'result': 'ok', '_images': [{'image_data_url': 'data:image/jpeg;base64," + ("A" * 2048) + "'}]}",
            tool_name="vector_search",
            arguments={"search_string": "test", "top_k": 5},
            for_whom="assistant",
            result_dict={
                "result": {"status": "ok"},
                "_images": [{"image_data_url": "data:image/jpeg;base64," + ("A" * 2048)}],
            },
        )

        rendered = msg.render(include={"role", "content"})
        self.assertIsInstance(rendered["content"], str)
        self.assertNotIn("data:image", rendered["content"])
        self.assertIn("redacted", rendered["content"].lower())

    def test_call_tool_sanitizes_private_keys_and_data_urls_in_content(self):
        llm = _FakeLLMInterface([])
        assistant_message = AssistantMessage(
            content="",
            stop_reason="tool_use",
            tool_call_id="tool_1",
            tool_call_name="_test_image_result_tool",
            tool_call_input={},
            tools=[_test_image_result_tool],
            tool_choice=ToolChoice(type="auto"),
        )

        tool_msg = llm.call_tool(assistant_message)
        self.assertIsInstance(tool_msg, ToolMessage)
        self.assertIn("result", tool_msg.content)
        self.assertIn("_image_instruction", tool_msg.content)
        self.assertNotIn("_images", tool_msg.content)
        self.assertNotIn("data:image", tool_msg.content)


class ToolImageMarkdownInjectionTests(SimpleTestCase):
    def test_complete_appends_image_markdown_when_missing_after_tool_result(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text="Here is the figure summary.",
                tool_call={},
                stop_reason='stop',
                input_usage=10,
                output_usage=20,
            ),
        ])
        convo = Conversation(
            system='You are a helpful assistant.',
            messages=[
                UserMessage(content='Show me the image result.'),
                ToolMessage(
                    content='{"result": {"type":"image"}}',
                    tool_name='vector_search',
                    arguments={"search_string": "figure", "top_k": 1},
                    for_whom='assistant',
                    result_dict={
                        "result": {
                            "[Result 1] -- Doc chunk #: 1": {
                                "type": "image",
                                "text": "Figure 2",
                                "image_url": "/aquillm/document_image/00000000-0000-0000-0000-000000000001/",
                            }
                        },
                        "_image_instruction": "Use markdown image syntax.",
                    },
                ),
            ],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 1024)

        self.assertEqual(changed, 'changed')
        self.assertIn("Here is the figure summary.", updated[-1].content)
        self.assertIn("![", updated[-1].content)
        self.assertIn("/aquillm/document_image/00000000-0000-0000-0000-000000000001/", updated[-1].content)

    def test_complete_appends_image_markdown_for_display_request_without_new_tool_call(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text="I can summarize what I found.",
                tool_call={},
                stop_reason='stop',
                input_usage=8,
                output_usage=16,
            ),
        ])
        convo = Conversation(
            system='You are a helpful assistant.',
            messages=[
                UserMessage(content='Find Figure 2'),
                ToolMessage(
                    content='{"result": {"type":"image"}}',
                    tool_name='vector_search',
                    arguments={"search_string": "Figure 2", "top_k": 5},
                    for_whom='assistant',
                    result_dict={
                        "result": {
                            "[Result 1] -- Paper chunk #: 12": {
                                "type": "image",
                                "text": "Figure 2",
                                "image_url": "/aquillm/document_image/00000000-0000-0000-0000-000000000002/",
                            }
                        },
                        "_image_instruction": "Use markdown image syntax.",
                    },
                ),
                AssistantMessage(content='Found the figure details.', stop_reason='stop', usage=20),
                UserMessage(content='Can you display it in chat?'),
            ],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 1024)

        self.assertEqual(changed, 'changed')
        self.assertIn("I can summarize what I found.", updated[-1].content)
        self.assertIn("![", updated[-1].content)
        self.assertIn("/aquillm/document_image/00000000-0000-0000-0000-000000000002/", updated[-1].content)


class CutoffContinuationTests(SimpleTestCase):
    def test_continues_cutoff_response_before_compact_fallback(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text='Scalable patent analysis outline:\n1. Ingestion and preprocessing',
                tool_call={},
                stop_reason='max_tokens',
                input_usage=5,
                output_usage=5,
            ),
            LLMResponse(
                text='2. Claim parsing and structured extraction.\n3. Prior-art ranking.\n4. Validation and reporting.',
                tool_call={},
                stop_reason='stop',
                input_usage=5,
                output_usage=5,
            ),
        ])
        convo = Conversation(
            system='You are a helpful assistant.',
            messages=[UserMessage(content='Give me a detailed patent-analysis implementation outline.')],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 2048)

        self.assertEqual(changed, 'changed')
        self.assertEqual(len(llm.calls), 2)
        self.assertIn('1. Ingestion and preprocessing', updated[-1].content)
        self.assertIn('2. Claim parsing and structured extraction.', updated[-1].content)


class MessageAdapterTests(TestCase):
    """Tests for converting between Pydantic messages and Django Message rows.

    Verifies that each message type (user, assistant, tool) converts correctly
    in both directions, and that role-specific fields don't bleed across types.
    """

    def setUp(self):
        # Every test needs a user and conversation to attach messages to
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt='You are a helpful assistant.',
        )

    # -- Pydantic to Django --

    def test_user_message_to_django(self):
        """User messages should only have the common fields populated."""
        msg = UserMessage(content='Hello there')
        db_msg = pydantic_message_to_django(msg, self.db_convo, seq_num=0)

        self.assertEqual(db_msg.role, 'user')
        self.assertEqual(db_msg.content, 'Hello there')
        self.assertEqual(db_msg.sequence_number, 0)
        self.assertEqual(db_msg.message_uuid, msg.message_uuid)
        self.assertIsNone(db_msg.rating)
        # Assistant-specific fields should be empty for a user message
        self.assertIsNone(db_msg.model)
        self.assertIsNone(db_msg.tool_call_name)

    def test_assistant_message_to_django(self):
        """Assistant messages should have model, stop_reason, and usage populated."""
        msg = AssistantMessage(
            content='Here is my response.',
            model='claude-3-7-sonnet-latest',
            stop_reason='end_turn',
            usage=500,
        )
        db_msg = pydantic_message_to_django(msg, self.db_convo, seq_num=1)

        self.assertEqual(db_msg.role, 'assistant')
        self.assertEqual(db_msg.content, 'Here is my response.')
        self.assertEqual(db_msg.model, 'claude-3-7-sonnet-latest')
        self.assertEqual(db_msg.stop_reason, 'end_turn')
        self.assertEqual(db_msg.usage, 500)
        # Tool-specific fields should be empty for an assistant message
        self.assertIsNone(db_msg.tool_name)
        self.assertIsNone(db_msg.for_whom)

    def test_assistant_tool_call_to_django(self):
        """When an assistant calls a tool, the tool_call fields should be populated."""
        msg = AssistantMessage(
            content='Let me search for that.',
            model='claude-3-7-sonnet-latest',
            stop_reason='tool_use',
            tool_call_id='toolu_123',
            tool_call_name='vector_search',
            tool_call_input={'search_string': 'test', 'top_k': 5},
            usage=300,
        )
        db_msg = pydantic_message_to_django(msg, self.db_convo, seq_num=1)

        self.assertEqual(db_msg.tool_call_id, 'toolu_123')
        self.assertEqual(db_msg.tool_call_name, 'vector_search')
        self.assertEqual(db_msg.tool_call_input, {'search_string': 'test', 'top_k': 5})

    def test_tool_message_to_django(self):
        """Tool messages should have tool_name, arguments, for_whom, and result_dict populated."""
        msg = ToolMessage(
            content='Search results here',
            tool_name='vector_search',
            arguments={'search_string': 'test', 'top_k': 5},
            for_whom='assistant',
            result_dict={'result': 'some data'},
        )
        db_msg = pydantic_message_to_django(msg, self.db_convo, seq_num=2)

        self.assertEqual(db_msg.role, 'tool')
        self.assertEqual(db_msg.tool_name, 'vector_search')
        self.assertEqual(db_msg.arguments, {'search_string': 'test', 'top_k': 5})
        self.assertEqual(db_msg.for_whom, 'assistant')
        self.assertEqual(db_msg.result_dict, {'result': 'some data'})

    # -- Django to Pydantic --

    def test_django_user_to_pydantic(self):
        """A Django user Message row should convert to a UserMessage Pydantic object."""
        db_msg = Message.objects.create(
            conversation=self.db_convo,
            role='user',
            content='Hello',
            sequence_number=0,
        )
        pydantic_msg = django_message_to_pydantic(db_msg)

        self.assertIsInstance(pydantic_msg, UserMessage)
        self.assertEqual(pydantic_msg.content, 'Hello')
        self.assertEqual(pydantic_msg.message_uuid, db_msg.message_uuid)

    def test_django_assistant_to_pydantic(self):
        """A Django assistant Message row should convert to an AssistantMessage Pydantic object."""
        db_msg = Message.objects.create(
            conversation=self.db_convo,
            role='assistant',
            content='My response',
            model='claude-3-7-sonnet-latest',
            stop_reason='end_turn',
            usage=1000,
            sequence_number=1,
        )
        pydantic_msg = django_message_to_pydantic(db_msg)

        self.assertIsInstance(pydantic_msg, AssistantMessage)
        self.assertEqual(pydantic_msg.model, 'claude-3-7-sonnet-latest')
        self.assertEqual(pydantic_msg.stop_reason, 'end_turn')
        self.assertEqual(pydantic_msg.usage, 1000)

    def test_django_tool_to_pydantic(self):
        """A Django tool Message row should convert to a ToolMessage Pydantic object."""
        db_msg = Message.objects.create(
            conversation=self.db_convo,
            role='tool',
            content='Results',
            tool_name='vector_search',
            for_whom='assistant',
            result_dict={'result': 'data'},
            sequence_number=2,
        )
        pydantic_msg = django_message_to_pydantic(db_msg)

        self.assertIsInstance(pydantic_msg, ToolMessage)
        self.assertEqual(pydantic_msg.tool_name, 'vector_search')
        self.assertEqual(pydantic_msg.for_whom, 'assistant')
        self.assertEqual(pydantic_msg.result_dict, {'result': 'data'})

    # -- Feedback text tests --

    def test_feedback_text_pydantic_to_django(self):
        """Feedback text should be saved to Django model."""
        msg = AssistantMessage(
            content='Here is my response.',
            model='claude-3-7-sonnet-latest',
            stop_reason='end_turn',
            rating=5,
            feedback_text='This was really helpful!',
        )
        db_msg = pydantic_message_to_django(msg, self.db_convo, seq_num=0)

        self.assertEqual(db_msg.feedback_text, 'This was really helpful!')

    def test_feedback_text_django_to_pydantic(self):
        """Feedback text should be loaded from Django model."""
        db_msg = Message.objects.create(
            conversation=self.db_convo,
            role='assistant',
            content='Test response',
            sequence_number=0,
            feedback_text='Great answer!',
        )

        pydantic_msg = django_message_to_pydantic(db_msg)

        self.assertEqual(pydantic_msg.feedback_text, 'Great answer!')

    def test_feedback_text_null_by_default(self):
        """Feedback text should be None when not provided."""
        msg = AssistantMessage(
            content='Response without feedback',
            model='claude-3-7-sonnet-latest',
            stop_reason='end_turn',
        )
        db_msg = pydantic_message_to_django(msg, self.db_convo, seq_num=0)

        self.assertIsNone(db_msg.feedback_text)


class SaveLoadConversationTests(TestCase):
    """Tests for saving and loading full conversations.

    These test the complete flow: Pydantic conversation -> save to DB -> load from DB -> Pydantic conversation.
    This is the same path that runs when consumers.py calls __save() and connect().
    """

    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt='You are a helpful assistant.',
        )

    def test_save_and_load_round_trip(self):
        """Save a conversation, load it back, and verify all data is intact."""
        convo = Conversation(
            system='Test system prompt',
            messages=[
                UserMessage(content='What is Django?'),
                AssistantMessage(
                    content='Django is a web framework.',
                    model='claude-3-7-sonnet-latest',
                    stop_reason='end_turn',
                    usage=200,
                ),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)
        loaded = load_conversation_from_db(self.db_convo)

        self.assertEqual(loaded.system, 'Test system prompt')
        self.assertEqual(len(loaded.messages), 2)
        self.assertIsInstance(loaded.messages[0], UserMessage)
        self.assertIsInstance(loaded.messages[1], AssistantMessage)
        self.assertEqual(loaded.messages[0].content, 'What is Django?')
        self.assertEqual(loaded.messages[1].content, 'Django is a web framework.')
        self.assertEqual(loaded.messages[1].model, 'claude-3-7-sonnet-latest')

    def test_save_creates_correct_number_of_rows(self):
        """Each message in the conversation should become one row in the Message table."""
        convo = Conversation(
            system='Test',
            messages=[
                UserMessage(content='Hi'),
                AssistantMessage(content='Hello', stop_reason='end_turn', usage=100),
                UserMessage(content='Follow up'),
                AssistantMessage(content='Response', stop_reason='end_turn', usage=200),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)

        self.assertEqual(self.db_convo.db_messages.count(), 4)

    def test_save_replaces_previous_messages(self):
        """Saving again should replace old messages, not duplicate them."""
        convo1 = Conversation(
            system='Test',
            messages=[UserMessage(content='First message')],
        )
        save_conversation_to_db(convo1, self.db_convo)
        self.assertEqual(self.db_convo.db_messages.count(), 1)

        # Save again with an additional message — should be 2 total, not 3
        convo2 = Conversation(
            system='Test',
            messages=[
                UserMessage(content='First message'),
                AssistantMessage(content='Reply', stop_reason='end_turn', usage=100),
            ],
        )
        save_conversation_to_db(convo2, self.db_convo)
        self.assertEqual(self.db_convo.db_messages.count(), 2)

    def test_save_updates_system_prompt(self):
        """The system prompt on WSConversation should update when saving."""
        convo = Conversation(system='New system prompt', messages=[])
        save_conversation_to_db(convo, self.db_convo)

        self.db_convo.refresh_from_db()
        self.assertEqual(self.db_convo.system_prompt, 'New system prompt')

    def test_message_ordering_by_sequence_number(self):
        """Messages should load in the same order they were saved."""
        convo = Conversation(
            system='Test',
            messages=[
                UserMessage(content='First'),
                AssistantMessage(content='Second', stop_reason='end_turn', usage=100),
                UserMessage(content='Third'),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)
        loaded = load_conversation_from_db(self.db_convo)

        self.assertEqual(loaded.messages[0].content, 'First')
        self.assertEqual(loaded.messages[1].content, 'Second')
        self.assertEqual(loaded.messages[2].content, 'Third')


class RatingTests(TestCase):
    """Tests for message rating persistence.

    Ratings are updated via a direct single-row DB update (not the full delete+recreate save),
    so these tests verify that path works correctly.
    """

    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt='Test',
        )

    def test_rating_persists_through_save_and_load(self):
        """A rating set on a Pydantic message should survive the save/load round-trip."""
        convo = Conversation(
            system='Test',
            messages=[
                UserMessage(content='Hi'),
                AssistantMessage(
                    content='Hello',
                    stop_reason='end_turn',
                    usage=100,
                    rating=5,
                ),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)
        loaded = load_conversation_from_db(self.db_convo)

        self.assertEqual(loaded.messages[1].rating, 5)

    def test_rating_update_via_queryset(self):
        """Tests the same pattern used by rate() in consumers.py — a direct single-row update."""
        msg_uuid = uuid4()
        Message.objects.create(
            conversation=self.db_convo,
            role='assistant',
            content='Response',
            stop_reason='end_turn',
            message_uuid=msg_uuid,
            sequence_number=0,
        )

        # This is exactly how consumers.py updates a rating
        self.db_convo.db_messages.filter(message_uuid=msg_uuid).update(rating=4)

        msg = Message.objects.get(message_uuid=msg_uuid)
        self.assertEqual(msg.rating, 4)

    def test_rating_change(self):
        """A rating should be changeable from one value to another."""
        msg_uuid = uuid4()
        Message.objects.create(
            conversation=self.db_convo,
            role='assistant',
            content='Response',
            stop_reason='end_turn',
            message_uuid=msg_uuid,
            sequence_number=0,
            rating=3,
        )

        self.db_convo.db_messages.filter(message_uuid=msg_uuid).update(rating=1)

        msg = Message.objects.get(message_uuid=msg_uuid)
        self.assertEqual(msg.rating, 1)


class BuildFrontendJsonTests(TestCase):
    """Tests for the JSON structure sent to the frontend over WebSocket.

    The frontend expects a specific format — these tests make sure
    build_frontend_conversation_json() produces exactly that structure.
    """

    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt='You are a helpful assistant.',
        )

    def test_basic_structure(self):
        """Output should have 'system' and 'messages' keys with correct values."""
        Message.objects.create(
            conversation=self.db_convo,
            role='user',
            content='Hello',
            sequence_number=0,
        )

        result = build_frontend_conversation_json(self.db_convo)

        self.assertEqual(result['system'], 'You are a helpful assistant.')
        self.assertEqual(len(result['messages']), 1)
        self.assertEqual(result['messages'][0]['role'], 'user')
        self.assertEqual(result['messages'][0]['content'], 'Hello')
        self.assertIn('message_uuid', result['messages'][0])

    def test_assistant_with_tool_call(self):
        """Assistant messages with tool calls should include tool_call fields and usage."""
        Message.objects.create(
            conversation=self.db_convo,
            role='assistant',
            content='Let me search.',
            tool_call_name='vector_search',
            tool_call_input={'search_string': 'test'},
            usage=300,
            sequence_number=0,
        )

        result = build_frontend_conversation_json(self.db_convo)
        msg = result['messages'][0]

        self.assertEqual(msg['tool_call_name'], 'vector_search')
        self.assertEqual(msg['tool_call_input'], {'search_string': 'test'})
        self.assertEqual(msg['usage'], 300)

    def test_tool_message(self):
        """Tool messages should include tool_name, for_whom, and result_dict."""
        Message.objects.create(
            conversation=self.db_convo,
            role='tool',
            content='Results',
            tool_name='vector_search',
            for_whom='assistant',
            result_dict={'result': 'data'},
            sequence_number=0,
        )

        result = build_frontend_conversation_json(self.db_convo)
        msg = result['messages'][0]

        self.assertEqual(msg['tool_name'], 'vector_search')
        self.assertEqual(msg['for_whom'], 'assistant')
        self.assertEqual(msg['result_dict'], {'result': 'data'})

    def test_message_uuid_is_string(self):
        """UUID should be serialized as a string for JSON compatibility."""
        Message.objects.create(
            conversation=self.db_convo,
            role='user',
            content='Hi',
            sequence_number=0,
        )

        result = build_frontend_conversation_json(self.db_convo)

        self.assertIsInstance(result['messages'][0]['message_uuid'], str)


class ConversationTitleTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='title-user', password='testpass')
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt='You are a helpful assistant.',
        )
        Message.objects.create(
            conversation=self.db_convo,
            role='user',
            content='Plan a 3 day Yosemite itinerary',
            sequence_number=0,
        )
        Message.objects.create(
            conversation=self.db_convo,
            role='assistant',
            content='Sure, here is an itinerary you can use.',
            stop_reason='end_turn',
            sequence_number=1,
        )

    def test_set_name_falls_back_when_llm_returns_generic_title(self):
        fake_llm = _FakeTitleLLM('conversation')
        fake_config = SimpleNamespace(llm_interface=fake_llm)
        with patch('aquillm.models.apps.get_app_config', return_value=fake_config):
            self.db_convo.set_name()

        self.db_convo.refresh_from_db()
        self.assertEqual(self.db_convo.name, 'Plan a 3 day Yosemite itinerary')

    def test_set_name_cleans_wrapped_llm_title(self):
        fake_llm = _FakeTitleLLM('  "*Yosemite Road Trip Plan*"  ')
        fake_config = SimpleNamespace(llm_interface=fake_llm)
        with patch('aquillm.models.apps.get_app_config', return_value=fake_config):
            self.db_convo.set_name()

        self.db_convo.refresh_from_db()
        self.assertEqual(self.db_convo.name, 'Yosemite Road Trip Plan')
